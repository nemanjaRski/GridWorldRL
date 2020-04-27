import numpy as np
import random as rand


class ExperienceBuffer:

    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, episode_experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(episode_experience)

    def sample(self, batch_size, trace_length):

        # take batch_size of episodes from the buffer
        sampled_episodes = rand.sample(self.buffer, batch_size)
        sampled_traces = []

        # from random starting point take next "trace_length" items from each episode
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return sampled_traces
