from gridworld import GameEnv
from agent import DRQNAgent
from networks import Networks

import numpy as np
import time

def log_game(print_every, losses, num_episode, rewards, prob_random):

    mean_loss = np.mean(losses[-print_every:])
    mean_reward = np.mean(rewards[-print_every:])
    current_time = time.strftime("%H:%M:%S", time.localtime())

    print("Time: {} Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                current_time, num_episode, mean_reward, prob_random, mean_loss))


def train():

    env = GameEnv(partial=False, size=5, num_goals=4, num_fires=2)

    action_size = env.actions # 4
    state_size = env.reset().shape # 84,84,3
    trace_length = 8
    learning_rate = 0.00025
    final_layer_size = 64

    explore_episodes = 200

    max_episodes = 20000
    curr_episode = 0

    print_every = 100
    update_frequency = 5

    agent = DRQNAgent(state_size, action_size, trace_length)

    agent.main_model = Networks.drqn(trace_length, state_size, action_size, final_layer_size, learning_rate)
    agent.target_model = Networks.drqn(trace_length, state_size, action_size, final_layer_size, learning_rate)

    #Game setup
    total_steps = 0
    max_steps = 50

    rewards = []
    losses = []

    empty_frames = []
    empty_state = np.zeros((84, 84, 3))

    for n in range(trace_length-1):
        empty_frames.append([empty_state, -1, 0, empty_state])

    while curr_episode < max_episodes:

        curr_episode += 1
        state = env.reset()

        done = False

        curr_step = 0

        episode_reward = 0
        episode_buffer = empty_frames
        episode_buffer.append([state, -1, 0, empty_state])

        while curr_step < max_steps and not done:

            curr_step += 1

            state_series = np.array([trace[0] for trace in episode_buffer[-agent.trace_length:]])
            action_idx = agent.get_action(state_series)
            next_state, reward, done = env.step(action_idx)

            episode_buffer.append([state, reward, done, next_state])

            episode_reward += reward

            state = next_state

        if agent.epsilon > agent.epsilon_end:
            agent.epsilon -= (agent.epsilon_start - agent.epsilon_end) / agent.explore

        if curr_episode % update_frequency == 0 and curr_episode > explore_episodes:
            q_max, loss = agent.train()
            losses.append(loss)

        rewards.append(episode_reward)

        agent.memory.add(episode_buffer)
        total_steps += curr_step

        if curr_episode % print_every == 0 and curr_episode > explore_episodes:
            log_game(print_every, losses, curr_episode, rewards, agent.epsilon)


if __name__ == '__main__':
    train()
