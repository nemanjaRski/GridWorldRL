import numpy as np
import random
from sum_tree import SumTree


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

        return np.reshape(sampled_traces, [sample_batch_size * sample_trace_length, 6])


class prioritized_experience_buffer:
    def __init__(self, buffer_size=5, epsilon=0.01, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001,
                 abs_err_upper=1., max_episode_length=50, sample_trace_length=8):
        self.buffer = np.zeros(buffer_size, dtype=object)
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper
        self.sample_trace_length = sample_trace_length
        self.tree = SumTree(buffer_size * (max_episode_length + 1 - self.sample_trace_length))
        self.buffer_pointer = 0

    def add(self, experience):
        self.buffer[self.buffer_pointer] = experience

        for i in range(0, len(experience) - self.sample_trace_length + 1):
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0:
                max_p = self.abs_err_upper
            self.tree.add(max_p, [self.buffer_pointer, i])

        self.buffer_pointer += 1
        if self.buffer_pointer >= self.buffer_size:  # replace when exceed the capacity
            self.buffer_pointer = 0

    def sample_from_tree(self, sample_batch_size):
        b_idx = np.empty((sample_batch_size,), dtype=np.int32)
        b_memory = np.empty((sample_batch_size, len(self.tree.data[0])))
        ISWeights = np.empty((sample_batch_size, 1))

        pri_seg = self.tree.total_p / sample_batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p

        for i in range(sample_batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def sample(self, sample_batch_size):
        tree_index, buffer_indexes, ISWeights = self.sample_from_tree(sample_batch_size)

        episode_indexes = [int(x[0]) for x in buffer_indexes]
        step_indexes = [int(x[1]) for x in buffer_indexes]
        sampled_episodes = self.buffer[episode_indexes]
        sampled_traces = []
        for idx, episode in enumerate(sampled_episodes):
            point = step_indexes[idx]
            sampled_traces.append(episode[point:point + self.sample_trace_length])
        sampled_traces = np.array(sampled_traces)

        return np.reshape(sampled_traces, [sample_batch_size * self.sample_trace_length, 6]), tree_index, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        error_counter = 0
        for index in tree_idx:
            for i in range(index,  index + self.sample_trace_length):
                self.tree.update(i, ps[error_counter])
                error_counter += 1
                if (i + 1) % 43 == 42:
                    error_counter += index + self.sample_trace_length - i - 1
                    break
