import random
import os
import tf_slim as slim
from gridworld import gameEnv
from helpers import *
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

env = gameEnv(partial=True, size=9, num_goals=4, num_fires=2)


class Qnetwork:
    def __init__(self, num_features, rnn_cell, scope_name, num_actions):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalar_input = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.image_in = tf.reshape(self.scalar_input, shape=[-1, 84, 84, 3])
        self.conv1 = slim.convolution2d(
            inputs=self.image_in, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv1')
        self.conv2 = slim.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv2')
        self.conv3 = slim.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv3')
        self.conv4 = slim.convolution2d(
            inputs=self.conv3, num_outputs=num_features,
            kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=scope_name + '_conv4')

        self.train_length = tf.placeholder(dtype=tf.int32)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.conv_flat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.train_length, num_features])
        self.rnn_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.conv_flat, cell=rnn_cell, dtype=tf.float32, initial_state=self.rnn_state_in,
            scope=scope_name + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, num_features])
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.stream_advantage, self.stream_value = tf.split(self.rnn, 2, 1)
        self.advantage_weights = tf.Variable(tf.random_normal([num_features // 2, num_actions]))
        self.value_weights = tf.Variable(tf.random_normal([num_features // 2, 1]))
        self.advantage = tf.matmul(self.stream_advantage, self.advantage_weights)
        self.value = tf.matmul(self.stream_value, self.value_weights)

        self.salience = tf.gradients(self.advantage, self.image_in)
        # Then combine them together to get our final Q-values.
        self.q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.q_out, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_one_hot), axis=1)

        self.td_error = tf.square(self.target_q - self.q)

        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.mask_a = tf.zeros([self.batch_size, self.train_length // 2])
        self.mask_b = tf.ones([self.batch_size, self.train_length // 2])
        self.mask = tf.concat([self.mask_a, self.mask_b], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_model = self.trainer.minimize(self.loss)


class experience_buffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, sample_batch_size, sample_trace_length):
        sampled_episodes = random.sample(self.buffer, sample_batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - sample_trace_length)
            sampled_traces.append(episode[point:point + sample_trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [sample_batch_size * sample_trace_length, 5])


# Setting the training parameters
batch_size = 4  # How many experience traces to use for each training step.
trace_length = 8  # How long each experience trace will be when training
update_freq = 5  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
start_exploration = 1  # Starting chance of random action
end_exploration = 0.1  # Final chance of random action
anneling_steps = 10000  # How many steps of training to reduce startE to endE.
num_episodes = 100  # How many episodes of game environment to train network with.
pre_train_steps = 1000  # How many steps of random actions before training begins.
load_model = False  # Whether to load a saved model.
path = "./drqn"  # The path to save our model to.
final_layer_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
max_ep_length = 50  # The max allowed length of our episode.
time_per_step = 1  # Length of each step used in gif creation
print_freq = 10  # Number of epidoes to periodically save for analysis
save_gif_freq = 2000
save_model_freq = 1000
tau = 0.001

tf.reset_default_graph()
# We define the cells for the primary and target q-networks
rnn_cell_main = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=final_layer_size, state_is_tuple=True)
rnn_cell_target = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=final_layer_size, state_is_tuple=True)
main_q_network = Qnetwork(final_layer_size, rnn_cell_main, 'main', 4)
target_q_network = Qnetwork(final_layer_size, rnn_cell_target, 'target', 4)

init = tf.global_variables_initializer()

model_saver = tf.train.Saver(max_to_keep=5)

trainables = tf.trainable_variables()

target_ops = update_target_graph(trainables, tau)

exp_buffer = experience_buffer()

# Set the rate of random action decrease.
exploration = start_exploration
exploration_drop_rate = (start_exploration - end_exploration) / anneling_steps

# create lists to contain total rewards and steps per episode
steps_list = []
rewards_list = []
red_list = []
green_list = []
stuck_list = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

##Write the first line of the master log-file for the Control Center

with open('./Center/log.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

with tf.Session() as sess:
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        model_saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)

    update_target(target_ops, sess)  # Set the target network to be equal to the primary network.
    for current_episode in range(num_episodes):
        episode_buffer = []
        # Reset environment and get first new observation
        state = env.reset()
        state_processed = process_state(state)
        done = False
        episode_reward = 0
        number_of_red = 0
        number_of_green = 0
        number_of_stuck = 0
        current_step = 0
        previous_rnn_state = (np.zeros([1, final_layer_size]), np.zeros([1, final_layer_size]))  # Reset the recurrent layer's hidden state
        # The Q-Network
        while current_step < max_ep_length:
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < exploration or total_steps < pre_train_steps:
                next_rnn_state = sess.run(main_q_network.rnn_state,
                                          feed_dict={main_q_network.scalar_input: [state_processed / 255.0], main_q_network.train_length: 1,
                                                     main_q_network.rnn_state_in: previous_rnn_state, main_q_network.batch_size: 1})
                action = np.random.randint(0, 4)
            else:
                action, next_rnn_state = sess.run([main_q_network.predict, main_q_network.rnn_state],
                                                  feed_dict={main_q_network.scalar_input: [state_processed / 255.0], main_q_network.train_length: 1,
                                                             main_q_network.rnn_state_in: previous_rnn_state, main_q_network.batch_size: 1})
                action = action[0]
            next_state, reward, done = env.step(action)
            next_state_processed = process_state(next_state)
            total_steps += 1
            episode_buffer.append(np.reshape(np.array([state_processed, action, reward, next_state_processed, done]), [1, 5]))
            if total_steps > pre_train_steps:
                if exploration > end_exploration:
                    exploration -= exploration_drop_rate

                if total_steps % update_freq == 0:
                    update_target(target_ops, sess)
                    # Reset the recurrent layer's hidden state
                    rnn_state_train = (np.zeros([batch_size, final_layer_size]), np.zeros([batch_size, final_layer_size]))

                    train_batch = exp_buffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    main_actions = sess.run(main_q_network.predict, feed_dict={
                        main_q_network.scalar_input: np.vstack(train_batch[:, 3] / 255.0),
                        main_q_network.train_length: trace_length, main_q_network.rnn_state_in: rnn_state_train, main_q_network.batch_size: batch_size})
                    target_q_values = sess.run(target_q_network.q_out, feed_dict={
                        target_q_network.scalar_input: np.vstack(train_batch[:, 3] / 255.0),
                        target_q_network.train_length: trace_length, target_q_network.rnn_state_in: rnn_state_train,
                        target_q_network.batch_size: batch_size})
                    end_multiplier = -(train_batch[:, 4] - 1)
                    double_q = target_q_values[range(batch_size * trace_length), main_actions]
                    target_q = train_batch[:, 2] + (y * double_q * end_multiplier)
                    # Update the network with our target values.
                    sess.run(main_q_network.update_model,
                             feed_dict={main_q_network.scalar_input: np.vstack(train_batch[:, 0] / 255.0),
                                        main_q_network.target_q: target_q,
                                        main_q_network.actions: train_batch[:, 1], main_q_network.train_length: trace_length,
                                        main_q_network.rnn_state_in: rnn_state_train, main_q_network.batch_size: batch_size})
            episode_reward += reward
            if reward == 1:
                number_of_green += 1
            elif reward == -1:
                number_of_red += 1
            elif reward < -1:
                number_of_stuck += 1
            state_processed = next_state_processed
            state = next_state
            previous_rnn_state = next_rnn_state
            current_step += 1
            if done:
                break

        # Add the episode to the experience buffer
        buffer_array = np.array(episode_buffer)
        episode_buffer = list(zip(buffer_array))
        exp_buffer.add(episode_buffer)
        steps_list.append(current_step)
        rewards_list.append(episode_reward)
        red_list.append(number_of_red)
        green_list.append(number_of_green)
        stuck_list.append(number_of_stuck)

        # Periodically save the model.
        if current_episode % save_model_freq == 0 and current_episode != 0:
            model_saver.save(sess, path + '/model-' + str(current_episode) + '.cptk')
            print("Saved Model")
        if len(rewards_list) % print_freq == 0 and len(rewards_list) != 0:
            log_game(print_freq, green_list, red_list, stuck_list, current_episode + 1, rewards_list, exploration)
        if len(rewards_list) % save_gif_freq == 0 and len(rewards_list) != 0:
            save_to_center(current_episode, rewards_list, steps_list, np.reshape(np.array(episode_buffer), [len(episode_buffer), 5]), \
                           print_freq, final_layer_size, sess, main_q_network, time_per_step)
    model_saver.save(sess, path + '/model-' + str(current_episode) + '.cptk')
