from __future__ import division
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Lambda, MaxPooling2D
import keras.backend as K
from keras.optimizers import Adam
import os
import time
import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')




class Qnetwork():
    def __init__(self, env, final_layer_size, lr):

        state_shape = env.reset().shape
        num_actions = env.action_space.n

        self.inputs = Input(shape=state_shape, name="main_input")

        self.model = Dense(final_layer_size, activation="relu", use_bias=False, name="hidden_layer_1")(self.inputs)
        self.model = Dense(num_actions, activation="relu", use_bias=False, name="hidden_layer_2")(self.model)
        self.model = Model(self.inputs, self.model)
        self.model.compile(optimizer=Adam(lr), loss="mse")


def update_target_graph(main_graph, target_graph, tau):
    update_weights = (np.array(main_graph.get_weights()) * tau) + (np.array(target_graph.get_weights()) * (1 - tau))
    target_graph.set_weights(update_weights)

def log_game(print_every, losses, num_epochs, total_steps, num_episode, rewards, prob_random):

    mean_loss = np.mean(losses[-(print_every * num_epochs):])
    current_time = time.strftime("%H:%M:%S", time.localtime())

    print("Time: {} Total steps: {} Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
        current_time, total_steps, num_episode, np.mean(rewards[-print_every:]), prob_random, mean_loss))


class ExpierienceReplay:
    def __init__(self, buffer_size=20000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.extend(experience)
        self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, size):
        sample_idxs = np.random.randint(len(self.buffer), size=size)
        sample_output = [self.buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output, (size, -1))
        return sample_output


# hyperparameters
batch_size = 32
num_epochs = 20
update_freq = 5
y = 0.95
tau = 0.01
learning_rate = 0.01
prob_random_start = 1
prob_random_end = 0.01
annealing_steps = 1000.
final_layer_size = 64
num_episodes = 100000
pre_train_episodes = 200
max_num_step = 200
goal = -150
state_dimension = len(env.reset())

# save/load weights
load_model = False
path = "./models/mountain_car"
conv_weights_file = path + "/conv_weights.h5"
main_weights_file = path + "/main_weights.h5"
target_weights_file = path + "/target_weights.h5"
print_every = 100
save_every = 1000

K.clear_session()

main_qn = Qnetwork(env, final_layer_size, learning_rate)
target_qn = Qnetwork(env, final_layer_size, learning_rate)
main_qn.model.summary()

update_target_graph(main_qn.model, target_qn.model, 1)

experience_replay = ExpierienceReplay()

prob_random = prob_random_start
prob_random_drop = (prob_random_start - prob_random_end) / annealing_steps

num_steps = []
rewards = []
losses = [0]
total_steps = 0
num_episode = 0

if not os.path.exists(path):
    os.makedirs(path)

if load_model is True:
    if os.path.exists(main_weights_file):
        print("Loading main weights")
        main_qn.model.load_weights(main_weights_file)
    if os.path.exists(target_weights_file):
        print("Loading target weights")
        target_qn.model.load_weights(target_weights_file)


finished_a_game = False

max_height = -10
while num_episode < num_episodes:

    episode_buffer = ExpierienceReplay()

    state = env.reset()
    done = False
    sum_rewards = 0
    cur_step = 0
    action = 0

    action_freq = 50 - num_episode // 100
    if action_freq < 5 and num_episode < 20000:
        action_freq = 5
    elif action_freq < 1:
        action_freq = 1

    while cur_step < max_num_step and not done:
        cur_step += 1
        total_steps += 1

        if cur_step % action_freq == 0:
            if np.random.rand() < prob_random or pre_train_episodes < num_episode:
                action = np.random.randint(env.action_space.n)
            else:
                state = np.reshape(state, [1, state_dimension])
                q_values = main_qn.model.predict([state])
                action = np.argmax(q_values)

        next_state, reward, done, _ = env.step(action)

        if cur_step % 20 == 0:
            episode = np.array([state, action, reward, next_state, done])
            episode = episode.reshape(1, -1)
            episode_buffer.add(episode)

        sum_rewards += reward
        state = next_state

    if num_episode > pre_train_episodes:
        if prob_random > prob_random_end:
            prob_random -= prob_random_drop

        if num_episode % update_freq == 0:
            for num_epoch in range(num_epochs):
                train_batch = experience_replay.sample(batch_size)
                train_state, train_action, train_reward, train_next_state, train_done = train_batch.T



                train_action = train_action.astype(np.int)

                train_state = np.vstack(train_state)
                train_next_state = np.vstack(train_next_state)

                target_q = target_qn.model.predict(train_state)
                target_q_next_state = target_qn.model.predict(train_next_state)

                main_q_action = main_qn.model.predict(train_next_state)
                train_next_state_action = np.argmax(main_q_action, axis=1)
                train_next_state_action = train_next_state_action.astype(np.int)

                train_gameover = train_done == 0

                train_next_state_values = target_q_next_state[range(batch_size), train_next_state_action]

                actual_reward = train_reward + (y * train_next_state_values * train_gameover)
                target_q[range(batch_size), train_action] = actual_reward

                loss = main_qn.model.train_on_batch(train_state, target_q)
                losses.append(loss)

            update_target_graph(main_qn.model, target_qn.model, tau)

        if (num_episode + 1) % save_every == 0:
            main_qn.model.save_weights(main_weights_file)
            target_qn.model.save_weights(target_weights_file)

    num_episode += 1

    experience_replay.add(episode_buffer.buffer)
    num_steps.append(cur_step)
    rewards.append(sum_rewards)

    if num_episode % print_every == 0:
        log_game(print_every, losses, num_epochs, total_steps, num_episode, rewards, prob_random)


    if np.mean(rewards[-print_every:]) >= goal and num_episode > 1000:
        print("Training complete!")
        break

main_qn.model.save_weights(main_weights_file)
target_qn.model.save_weights(target_weights_file)
