import numpy as np
import random


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
