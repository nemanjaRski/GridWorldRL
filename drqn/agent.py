from experience_buffer import ExperienceBuffer

import numpy as np
import random as rand


class DRQNAgent:

    def __init__(self, state_size, action_size, trace_length):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.tau = 0.1

        self.epsilon_start = 0.2
        self.epsilon_end = 0.1
        self.epsilon = 0.2

        self.explore = 10000

        self.batch_size = 32
        self.num_epochs = 20
        self.trace_length = trace_length

        self.memory = ExperienceBuffer()

        self.main_model = None
        self.target_model = None

    def update_target_model(self, rewrite=False):

        updated_weights = None

        if rewrite is False:
            main_weights = self.main_model.get_weights() * self.tau
            target_weights = self.target_model.get_weights() * (1 - self.tau)
            updated_weights = np.array(main_weights + target_weights)
        else:
            updated_weights = self.main_model.get_weights()

        self.target_model.set_weights(updated_weights)

    def get_action(self, state):

        if np.random.rand() < self.epsilon:
            action_idx = rand.randrange(self.action_size)
        else:
            q_values = self.main_model.predict(np.array([state]))
            action_idx = np.argmax(q_values)

        return action_idx


    def train(self):

        train_losses = []
        for num_epoch in range(self.num_epochs):

            sample_traces = self.memory.sample(self.batch_size, self.trace_length)

            state = np.zeros(((self.batch_size, self.trace_length) + self.state_size))
            next_state = np.zeros(((self.batch_size, self.trace_length) + self.state_size))

            action = np.zeros((self.batch_size, self.trace_length))
            reward = np.zeros((self.batch_size, self.trace_length))


            for episode in range(self.batch_size):
                for trace in range(self.trace_length):

                    state[episode, trace, :, :, :] = sample_traces[episode][trace][0]
                    action[episode, trace] = sample_traces[episode][trace][1]
                    reward[episode, trace] = sample_traces[episode][trace][2]
                    next_state[episode, trace, :, :, :] = sample_traces[episode][trace][3]

            target_q = self.main_model.predict(state)
            target_q_next = self.target_model.predict(next_state)

            for episode in range(self.batch_size):
                a = np.argmax(target_q_next[episode])
                target_q[episode][int(action[episode][-1])] = reward[episode][-1] + self.gamma * (target_q_next[episode][a])

            loss = self.main_model.train_on_batch(state, target_q)
            train_losses.append(loss)

        return np.max(target_q[-1, -1]), np.mean(train_losses)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

