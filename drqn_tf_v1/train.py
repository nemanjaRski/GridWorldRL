from network import Qnetwork
from experience_buffer import experience_buffer
from helpers import *
from ai_life import GameEnv
from action_selection_strategies import *
import os
import tensorflow.compat.v1 as tf

"""Game environment"""
env = GameEnv(partial=True, size=7, num_goals=4, num_fires=4, for_print=False, sight=2)
action_space_size = env.actions
state_shape = env.reset().shape

"""Training parameters"""
batch_size = 4
trace_length = 8
update_freq = 5
num_episodes = 100000
pre_train_steps = 250
max_ep_length = 50

"""Model parameters"""
final_layer_size = 512
learning_rate = 0.0001

"""Q value parameters"""
tau = 0.001
y = .95

"""Exploration parameters"""
exploration_start = 1
exploration_end = 0.1
exploration = exploration_start
exploration_drop_rate = (exploration_start - exploration_end) / ((num_episodes * max_ep_length - pre_train_steps) * 0.5)

"""Debug and save parameters"""
load_model = False
path_weights = "./drqn_weights"
path_results = "./drqn_train_results"
time_per_step = 1  # Length of each step used in gif creation
print_freq = 1
save_gif_freq = 1000
save_model_freq = 1000

"""Initializing buffers"""
exp_buffer = experience_buffer()
steps_list = []  # keeps track of steps per episode
rewards_list = []  # keeps track of rewards per episode
red_list = []  # keeps track of red items collected per episode
green_list = []  # keeps track of green items collected per episode
stuck_list = []  # keeps track of number of times agent didn't move per episode

tf.reset_default_graph()

"""Create RNN cells"""
rnn_cell_main = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=final_layer_size, state_is_tuple=True)
rnn_cell_target = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=final_layer_size, state_is_tuple=True)

"""Create main and target network"""
main_q_network = Qnetwork(final_layer_size, rnn_cell_main, 'main', action_space_size, state_shape, learning_rate)
target_q_network = Qnetwork(final_layer_size, rnn_cell_target, 'target', action_space_size, state_shape, learning_rate)

init = tf.global_variables_initializer()

model_saver = tf.train.Saver(max_to_keep=5)

"""Init target network"""
trainables = tf.trainable_variables()
target_ops = update_target_graph(trainables, tau)

if not os.path.exists(path_weights):
    os.makedirs(path_weights)

if not os.path.exists(path_results):
    os.makedirs(path_results)

with open(f"{path_results}/log.csv", 'w') as log_file:
    wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

with tf.Session() as sess:
    if load_model:
        print('Loading Model...')
        check_point = tf.train.get_checkpoint_state(path_weights)
        model_saver.restore(sess, check_point.model_checkpoint_path)
    sess.run(init)
    total_steps = 0
    current_episode = 1
    update_target(target_ops, sess)  # Set the target network to be equal to the primary network.
    while current_episode <= num_episodes:
        episode_buffer = []
        state = env.reset()
        done = False
        episode_reward = 0
        num_of_red = 0
        num_of_green = 0
        num_of_stuck = 0
        current_step = 0
        previous_rnn_state = (
            np.zeros([1, final_layer_size]),
            np.zeros([1, final_layer_size]))  # Reset the recurrent layer's hidden state
        previous_action = -1
        while current_step < max_ep_length:
            if total_steps < pre_train_steps:
                next_rnn_state = previous_rnn_state
                action = random_predict(np.ones(action_space_size), 0)
            else:
                q_out, next_rnn_state = sess.run([main_q_network.q_out, main_q_network.rnn_state],
                                                 feed_dict={main_q_network.image_in: [state],
                                                            main_q_network.train_length: 1,
                                                            main_q_network.rnn_state_in: previous_rnn_state,
                                                            main_q_network.batch_size: 1,
                                                            main_q_network.action_in: [previous_action],
                                                            main_q_network.keep_per: (1 - exploration) + 0.1})
                action = boltzman_predict(q_out, exploration)
            next_state, reward, done = env.step(action)
            total_steps += 1
            episode_buffer.append(
                np.reshape(np.array([state, action, reward, next_state, done, previous_action]), [1, 6]))
            if total_steps > pre_train_steps:
                if exploration > exploration_end:
                    exploration -= exploration_drop_rate

                if total_steps % update_freq == 0:
                    update_target(target_ops, sess)
                    # Reset the recurrent layer's hidden state
                    rnn_state_train = (
                        np.zeros([batch_size, final_layer_size]), np.zeros([batch_size, final_layer_size]))

                    train_batch = exp_buffer.sample(batch_size, trace_length)

                    main_actions = sess.run(main_q_network.predict, feed_dict={
                        main_q_network.image_in: np.array([*train_batch[:, 3]]),
                        main_q_network.train_length: trace_length, main_q_network.rnn_state_in: rnn_state_train,
                        main_q_network.batch_size: batch_size, main_q_network.action_in: train_batch[:, 1],
                        main_q_network.keep_per: 1.0})
                    target_q_values = sess.run(target_q_network.q_out, feed_dict={
                        target_q_network.image_in: np.array([*train_batch[:, 3]]),
                        target_q_network.train_length: trace_length, target_q_network.rnn_state_in: rnn_state_train,
                        target_q_network.batch_size: batch_size, target_q_network.action_in: train_batch[:, 1],
                        target_q_network.keep_per: 1.0})
                    end_multiplier = -(train_batch[:, 4] - 1)
                    double_q = target_q_values[range(batch_size * trace_length), main_actions]
                    target_q = train_batch[:, 2] + (y * double_q * end_multiplier)
                    # Update the network with our target values.
                    sess.run(main_q_network.update_model,
                             feed_dict={main_q_network.image_in: np.array([*train_batch[:, 0]]),
                                        main_q_network.target_q: target_q,
                                        main_q_network.actions: train_batch[:, 1],
                                        main_q_network.train_length: trace_length,
                                        main_q_network.rnn_state_in: rnn_state_train,
                                        main_q_network.batch_size: batch_size,
                                        main_q_network.action_in: train_batch[:, 5],
                                        main_q_network.keep_per: 1.0})
            episode_reward += reward

            num_of_green += int(reward == 1)
            num_of_red += int(reward == -1)
            num_of_stuck += int(reward < -1)
            state = next_state
            previous_rnn_state = next_rnn_state
            previous_action = action
            current_step += 1
            if done:
                break

        # Add the episode to the experience buffer
        buffer_array = np.array(episode_buffer)
        episode_buffer = list(zip(buffer_array))
        exp_buffer.add(episode_buffer)
        steps_list.append(current_step)
        rewards_list.append(episode_reward)
        red_list.append(num_of_red)
        green_list.append(num_of_green)
        stuck_list.append(num_of_stuck)

        # Periodically save the model.
        if current_episode % save_model_freq == 0:
            model_saver.save(sess, path_weights + '/model-' + str(current_episode) + '.cptk')
            print("Saved Model.")
        if current_episode % print_freq == 0:
            log_game(print_freq, green_list, red_list, stuck_list, current_episode, rewards_list, exploration)
        if current_episode % save_gif_freq == 0:
            save_to_center(current_episode, rewards_list, steps_list,
                           np.reshape(np.array(episode_buffer), [len(episode_buffer), 6]),
                           print_freq, final_layer_size, sess, main_q_network, time_per_step, path_results)
        current_episode += 1
    model_saver.save(sess, path_weights + '/model-' + str(current_episode) + '.cptk')
