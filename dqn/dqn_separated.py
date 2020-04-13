from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Flatten, Lambda, MaxPooling2D
import keras.backend as K
from keras.optimizers import Adam
import os
import time

from gridworld import gameEnv
import matplotlib.pyplot as plt
from coords import CoordinateChannel2D

env = gameEnv(partial=False, size=19, num_goals=15, num_fires=30)


def process_state(state):
    state = (state // 255).astype(np.uint8)
    return state


class ConvNetwork():
    def __init__(self, final_layer_size):

        self.inputs = Input(shape=[*process_state(env.state).shape], name="input")

        self.model = CoordinateChannel2D()(self.inputs)
        self.model = Conv2D(16, (3, 3), activation='relu', padding='same', name='enc_1')(self.model)
        self.model = MaxPooling2D((2, 2), padding='same', name='max_pool_1')(self.model)
        self.model = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_2')(self.model)
        self.model = MaxPooling2D((2, 2), padding='same', name='max_pool_2')(self.model)
        self.model = Conv2D(final_layer_size, (3, 3), activation='relu', padding='same', name='enc_3')(self.model)

        self.model = Model(self.inputs, self.model)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')

    def predict(self, state):
        return self.model.predict(np.array(state))


class QNetwork():

    def __init__(self, final_layer_size):

        self.inputs = Input(shape=[21, 21, final_layer_size], name='input')

        self.stream_AC = Lambda(lambda layer: layer[:, :, :, :final_layer_size // 2], name="advantage")(self.inputs)
        self.stream_VC = Lambda(lambda layer: layer[:, :, :, final_layer_size // 2:], name="value")(self.inputs)

        self.stream_AC = Flatten(name="advantage_flatten")(self.stream_AC)
        self.stream_VC = Flatten(name="value_flatten")(self.stream_VC)

        self.Advantage = Dense(env.actions, name="advantage_final")(self.stream_AC)
        self.Value = Dense(1, name="value_final")(self.stream_VC)

        self.model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1] - K.mean(val_adv[1],
                                                                                                  axis=1,
                                                                                                  keepdims=True))),
                            name="final_out")([self.Value, self.Advantage])

        self.model = Model(self.inputs, self.model)
        self.model.compile(optimizer=Adam(0.00025), loss="mse")

    def predict(self, state):
        return self.model.predict(np.array(state))


def update_target_graph(main_graph, target_graph, tau):
    update_weights = (np.array(main_graph.get_weights()) * tau) + (np.array(target_graph.get_weights()) * (1 - tau))
    target_graph.set_weights(update_weights)

def load_network_weights(neural_network, file_path, by_name = False):
    if os.path.exists(file_path):
        print("Loading weights from: " + file_path)
        neural_network.model.load_weights(main_weights_file, by_name=by_name)

class ExpierienceReplay():
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


def test(q_network, env, num_episodes):
    all_rewards = []
    for i in range(num_episodes):

        done = False
        num_step = 0
        state = env.reset()
        state = process_state(state)
        episode_reward = 0
        while not done and num_step < max_num_step:
            action = np.argmax(main_qn.model.predict(np.array([state])), axis=1)
            next_state, reward, done = env.step(action)
            if reward < -0:
                print("bad move")
            episode_reward += reward
            state = process_state(next_state)
            num_step += 1

        all_rewards.append(episode_reward)
    print(all_rewards)
    print(np.mean(all_rewards))


def log_game(print_every, losses, num_epochs, total_steps, num_episode, rewards,
             prob_random, elapsed_env_time, elapsed_train_time, steps_since_last_print):

    mean_loss = np.mean(losses[-(print_every * num_epochs):])
    current_time = time.strftime("%H:%M:%S", time.localtime())

    print("Time:{} Total steps:{} Episodes:{} Average Steps:{:0.2f} Mean reward:{:0.2f} Exploration:{:0.2f}, Loss:{:0.04f} Play:{:0.2f}s Train:{:0.2f}s".format(
        current_time, total_steps, num_episode, steps_since_last_print / print_every,np.mean(rewards[-print_every:]), prob_random, mean_loss,
        elapsed_env_time, elapsed_train_time))


def plot_game(q_network, env):

    f, axes = plt.subplots(nrows=max_num_step // 5, ncols=5, sharex=True, sharey=True, figsize=(25, 25))
    done = False
    num_step = 0
    sum_rewards = 0
    state = env.reset()

    while done is False and num_step < max_num_step:

        conv_state = conv_n.predict([state])
        q_values = q_network.model.predict(conv_state)
        action = np.argmax(q_values, axis=1)
        next_state, reward, done = env.step(action)
        sum_rewards += reward
        ax = axes.ravel()[num_step]
        ax.imshow(state)
        ax.set_axis_off()
        ax.set_title("|{:.1f} {:.1f} {:.1f} {:.1f} [{:.1f}]|".format(q_values[0][0], q_values[0][1], q_values[0][2], q_values[0][3], sum_rewards))
        state = next_state
        num_step += 1

    plt.show()

#hyperparameters
batch_size = 32
num_epochs = 20
update_freq = 5
y = 0.99
tau = 0.1
prob_random_start = 1
prob_random_end = 0.1
annealing_steps = 8000.
final_layer_size = 128
num_episodes = 10000
pre_train_episodes = 100
max_num_step = 50
goal = 20

#save/load weights
load_model = True
path = "./models"
main_weights_file = path + "/main_weights.h5"
target_weights_file = path + "/target_weights.h5"
conv_weights_file = path + "/conv_weights.h5"

print_every = 100
save_every = 1000

K.clear_session()

conv_n = ConvNetwork(final_layer_size)
main_qn = QNetwork(final_layer_size)
target_qn = QNetwork(final_layer_size)

conv_n.model.summary()
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

load_network_weights(conv_n, conv_weights_file, by_name=True)

if load_model is True:
    load_network_weights(main_qn, main_weights_file)
    load_network_weights(target_qn, target_weights_file)

elapsed_env_time = 0
elapsed_train_time = 0
steps_per_sec = 0
steps_since_last_print = 0
while num_episode < num_episodes:
    episode_buffer = ExpierienceReplay()

    state = env.reset()
    state = process_state(state)

    done = False
    sum_rewards = 0
    cur_step = 0


    start_env_time = time.time()
    while cur_step < max_num_step and not done:
        cur_step += 1
        total_steps += 1
        steps_since_last_print += 1
        if np.random.rand() < prob_random or num_episode < pre_train_episodes:
            action = np.random.randint(env.actions)
        else:
            conv_state = conv_n.predict([state])
            q_values = main_qn.predict(conv_state)
            action = np.argmax(q_values)

        next_state, reward, done = env.step(action)
        next_state = process_state(next_state)

        episode = np.array([[state], action, reward, [next_state], done])
        episode = episode.reshape(1, -1)
        episode_buffer.add(episode)
        sum_rewards += reward
        state = next_state
    elapsed_env_time += (time.time() - start_env_time)

    if num_episode > pre_train_episodes:
        if prob_random > prob_random_end:
            prob_random -= prob_random_drop

        train_batch = experience_replay.sample(batch_size)
        train_state, train_action, train_reward, train_next_state, train_done = train_batch.T

        start_train_time = time.time()
        if num_episode % update_freq == 0:

            for num_epoch in range(num_epochs):

                train_batch = experience_replay.sample(batch_size)
                train_state, train_action, train_reward, train_next_state, train_done = train_batch.T
                train_action = train_action.astype(np.int)

                
                # combined_train_states = np.vstack(np.append(train_state, train_next_state))
                # conv_combined_train_states = conv_n.predict(combined_train_states)
                # target_q_combined = target_qn.model.predict(conv_combined_train_states)
                # 
                # target_q = target_q_combined[:batch_size]
                # target_q_next_state = target_q_combined[batch_size:]
                # 
                # conv_train_state = conv_combined_train_states[:batch_size]
                # conv_train_next_state = conv_combined_train_states[batch_size:]
                

                train_state = np.vstack(train_state)
                conv_train_state = conv_n.predict(train_state)
                target_q = target_qn.model.predict(conv_train_state)

                train_next_state = np.vstack(train_next_state)
                conv_train_next_state = conv_n.predict(train_next_state)
                target_q_next_state = target_qn.model.predict(conv_train_next_state)

                main_q_values = main_qn.model.predict(conv_train_next_state)
                train_next_state_action = np.argmax(main_q_values, axis=1)
                train_next_state_action = train_next_state_action.astype(np.int)

                train_gameover = train_done == 0

                train_next_state_values = target_q_next_state[range(batch_size), train_next_state_action]

                actual_reward = train_reward + (y * train_next_state_values * train_gameover)
                target_q[range(batch_size), train_action] = actual_reward

                loss = main_qn.model.train_on_batch(conv_train_state, target_q)
                losses.append(loss)

            update_target_graph(main_qn.model, target_qn.model, tau)
        elapsed_train_time += (time.time() - start_train_time)
        if (num_episode + 1) % save_every == 0:
            main_qn.model.save_weights(main_weights_file)
            target_qn.model.save_weights(target_weights_file)

    num_episode += 1

    experience_replay.add(episode_buffer.buffer)
    num_steps.append(cur_step)
    rewards.append(sum_rewards)

    if num_episode % print_every == 0:

        log_game(print_every, losses, num_epochs, total_steps, num_episode, rewards,
                 prob_random, elapsed_env_time, elapsed_train_time, steps_since_last_print)
        steps_since_last_print = 0
        elapsed_env_time = 0
        elapsed_train_time = 0

    if np.mean(rewards[-print_every:]) >= goal:
        print("Training complete!")
        break

main_qn.model.save_weights(main_weights_file)
target_qn.model.save_weights(target_weights_file)

test(main_qn, env, 50)

plot_game(main_qn, env)



