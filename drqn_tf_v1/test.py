from network import Qnetwork
from helpers import *
from gridworld import GameEnv
import os
import tensorflow.compat.v1 as tf

"""Game environment"""
env = GameEnv(partial=True, size=19, num_goals=20, num_fires=15, for_print=True, sight=2)
action_space_size = env.actions
state_shape = env.reset().shape

num_episodes = 1000

path_weights = "./drqn_weights"
path_results = "./drqn_test_results"
final_layer_size = 512
learning_rate = 0.0001
max_ep_length = 50
time_per_step = 1
print_freq = 100
save_gif_freq = 100

tf.reset_default_graph()
rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=final_layer_size, state_is_tuple=True)
q_network = Qnetwork(final_layer_size, rnn_cell, 'main', action_space_size, state_shape, learning_rate)

model_saver = tf.train.Saver(max_to_keep=2)

# create lists to contain total rewards and steps per episode
steps_list = []  # keeps track of steps per episode
rewards_list = []  # keeps track of rewards per episode
red_list = []  # keeps track of red items collected per episode
green_list = []  # keeps track of green items collected per episode
stuck_list = []  # keeps track of number of times agent didn't move per episode
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path_results):
    os.makedirs(path_results)

with open(f"{path_results}/log.csv", 'w') as log_file:
    wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])


with tf.Session() as sess:


    check_point = tf.train.get_checkpoint_state(path_weights)
    model_saver.restore(sess, check_point.model_checkpoint_path)
    print('Model loaded...')
    current_episode = 1
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
        while current_step < max_ep_length:
            action, next_rnn_state = sess.run([q_network.predict, q_network.rnn_state],
                                              feed_dict={q_network.image_in: [state / 255.0],
                                                         q_network.train_length: 1,
                                                         q_network.rnn_state_in: previous_rnn_state,
                                                         q_network.batch_size: 1})
            action = action[0]
            next_state, reward, done = env.step(action)
            full_state = env.render_full_env()
            episode_buffer.append(
                np.reshape(np.array([state, action, reward, next_state, done, full_state]), [1, 6]))
            episode_reward += reward
            num_of_green += int(reward == 1)
            num_of_red += int(reward == -1)
            num_of_stuck += int(reward < -1)
            state = next_state
            previous_rnn_state = next_rnn_state
            current_step += 1
            if done:
                break

        # Add the episode to the experience buffer
        buffer_array = np.array(episode_buffer)
        episode_buffer = list(zip(buffer_array))
        steps_list.append(current_step)
        rewards_list.append(episode_reward)
        red_list.append(num_of_red)
        green_list.append(num_of_green)
        stuck_list.append(num_of_stuck)

        if current_episode % print_freq == 0:
            log_game(print_freq, green_list, red_list, stuck_list, current_episode, rewards_list, 0)
        if current_episode % save_gif_freq == 0:
            save_to_center(current_episode, rewards_list, steps_list,
                           np.reshape(np.array(episode_buffer), [len(episode_buffer), 6]),
                           print_freq, final_layer_size, sess, q_network, time_per_step, path_results, save_full_state=True)
        current_episode += 1

print(np.mean(rewards_list))
