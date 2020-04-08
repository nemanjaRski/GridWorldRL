from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Flatten, Lambda
import keras.backend as K
from keras.optimizers import Adam
import os

from gridworld import gameEnv

env = gameEnv(partial=False, size=5)


def process_state(state):
    return state


class Qnetwork():
    def __init__(self, final_layer_size):
        self.inputs = Input(shape=[*process_state(env.state).shape], name="main_input")

        self.model = Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation="relu",
            padding="valid",
            name="conv1"
        )(self.inputs)
        self.model = Conv2D(
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation="relu",
            padding="valid",
            name="conv2"
        )(self.model)
        self.model = Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=[1, 1],
            activation="relu",
            padding="valid",
            name="conv3"
        )(self.model)
        self.model = Conv2D(
            filters=final_layer_size,
            kernel_size=[7, 7],
            strides=[1, 1],
            activation="relu",
            padding="valid",
            name="conv4"
        )(self.model)

        self.stream_AC = Lambda(lambda layer: layer[:, :, :, :final_layer_size // 2], name="advantage")(self.model)
        self.stream_VC = Lambda(lambda layer: layer[:, :, :, final_layer_size // 2:], name="value")(self.model)

        self.stream_AC = Flatten(name="advantage_flatten")(self.stream_AC)
        self.stream_VC = Flatten(name="value_flatten")(self.stream_VC)

        self.Advantage = Dense(env.actions, name="advantage_final")(self.stream_AC)
        self.Value = Dense(1, name="value_final")(self.stream_VC)

        self.model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1] - K.mean(val_adv[1],
                                                                                                  axis=1,
                                                                                                  keepdims=True))),
                            name="final_out")([self.Value, self.Advantage])

        self.model = Model(self.inputs, self.model)
        self.model.compile(optimizer=Adam(0.0001), loss="mse")


def update_target_graph(main_graph, target_graph, tau):
    update_weights = (np.array(main_graph.get_weights()) * tau) + (np.array(target_graph.get_weights()) * (1 - tau))
    target_graph.set_weights(update_weights)


class ExpierienceReplay:
    def __init__(self, buffer_size=50000):
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


batch_size = 64
num_epochs = 20
update_freq = 5
y = 0.99
prob_random_start = 0.6
prob_random_end = 0.1
annealing_steps = 1000.
num_episodes = 10000
pre_train_episodes = 100
max_num_step = 50
load_model = False
path = "./models"
main_weights_file = path + "/main_weights.h5"
target_weights_file = path + "/target_weights.h5"

final_layer_size = 512

tau = 1
goal = 10

K.clear_session()

main_qn = Qnetwork(final_layer_size)
target_qn = Qnetwork(final_layer_size)

update_target_graph(main_qn.model, target_qn.model, 1)

experience_replay = ExpierienceReplay()

prob_random = prob_random_start
prob_random_drop = (prob_random_start - prob_random_end) / annealing_steps

num_steps = []
rewards = []
total_steps = 0

print_every = 50
save_every = 5

losses = [0]

num_episode = 0

if not os.path.exists(path):
    os.makedirs(path)
if load_model == True:
    if os.path.exists(main_weights_file):
        print("Loading main weights")
        main_qn.model.load_weights(main_weights_file)
    if os.path.exists(target_weights_file):
        print("Loading target weights")
        target_qn.model.load_weights(target_weights_file)

while num_episode < num_episodes:
    episode_buffer = ExpierienceReplay()

    state = env.reset()
    state = process_state(state)

    done = False
    sum_rewards = 0
    cur_step = 0

    while cur_step < max_num_step and not done:
        cur_step += 1
        total_steps += 1

        if np.random.rand() < prob_random or num_episode < pre_train_episodes:
            action = np.random.randint(env.actions)
        else:
            action = np.argmax(main_qn.model.predict(np.array([state])))

        next_state, reward, done = env.step(action)
        next_state = process_state(next_state)

        episode = np.array([[state], action, reward, [next_state], done])
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

                target_q_next_state = main_qn.model.predict(train_next_state)
                train_next_state_action = np.argmax(target_q_next_state, axis=1)
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
