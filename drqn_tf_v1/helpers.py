import numpy as np
import csv
import time
import os
import matplotlib.pyplot as plt


# This is a simple function to reshape our game frames.
def process_state(state1):
    return np.reshape(state1, [21168])


def log_game(print_every, green, red, stuck, num_episode, rewards, prob_random):
    mean_green = np.mean(green[-print_every:])
    mean_red = np.mean(red[-print_every:])
    mean_stuck = np.mean(stuck[-print_every:])
    mean_reward = np.mean(rewards[-print_every:])
    max_reward = np.max(rewards[-print_every:])
    min_reward = np.min(rewards[-print_every:])
    current_time = time.strftime("%H:%M:%S", time.localtime())

    print(
        "Time: {} Num episode: {} Mean reward: {:0.4f} Max reward: {:0.1f} Min reward: {:0.1f} Prob random: {:0.4f}, "
        "Green: {:0.04f} , Red: {:0.04f} , Stuck: {:0.04f}".format(
            current_time, num_episode, mean_reward, max_reward, min_reward, prob_random, mean_green, mean_red,
            mean_stuck))


def plot_hist(rewards, green, red, print_every, num_episode, base_path):
    rewards_hist = rewards[-print_every:]
    green_hist = green[-print_every:]
    red_hist = red[-print_every:]

    bins_reward = np.linspace(-10, 20, 30)
    bins_green_red = np.linspace(0, 20, 20)

    fig = plt.figure(figsize=(22, 4))
    subplot = fig.add_subplot(1, 3, 1)
    subplot.hist(rewards_hist, bins_reward, color='b', label='rewards')
    subplot = fig.add_subplot(1, 3, 2)
    subplot.hist(green_hist, bins_green_red, color='g', label='green')
    subplot = fig.add_subplot(1, 3, 3)
    subplot.hist(red_hist, bins_green_red, color='r', label='red')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    if not os.path.exists(f'{base_path}/plots'):
        os.makedirs(f'{base_path}/plots')

    fig.savefig(f'{base_path}/plots/hist_{num_episode}.png', bbox_inches="tight")


# These functions allows us to update the parameters of our target network with those of the primary network.
def update_target_graph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# Record performance metrics and episode logs for the Control Center.
def save_to_center(current_episode, reward_list, steps_list, episode_buffer, print_freq, time_per_step, path,
                   save_full_state=False):
    if not os.path.exists(f'{path}/frames'):
        os.makedirs(f'{path}/frames')

    images = list(episode_buffer[:, 0])
    images.append(episode_buffer[-1, 3])
    make_gif(np.array(images), f'{path}/frames/image' + str(current_episode) + '.gif',
             duration=len(images) * time_per_step,
             true_image=True, salience=False)
    if save_full_state:
        full_images = list(episode_buffer[:, 6])
        make_gif(np.array(full_images), f'{path}/frames/full_image' + str(current_episode) + '.gif',
                 duration=len(full_images) * time_per_step,
                 true_image=True, salience=False)

    log_to_csv(current_episode, reward_list, steps_list, episode_buffer, print_freq, path)


def log_to_csv(current_episode, reward_list, steps_list, episode_buffer, print_freq, path):
    with open(f"{path}/log.csv", 'a') as log_file:
        wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)
        wr.writerow([current_episode, np.mean(steps_list[-100:]), np.mean(reward_list[-print_freq:]),
                     './frames/image' + str(current_episode) + '.gif',
                     './frames/log' + str(current_episode) + '.csv'])
        log_file.close()
    with open(f'{path}/frames/log' + str(current_episode) + '.csv', 'w') as log_file:
        wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)
        wr.writerow(["ACTION", "REWARD", "A0", "A1", 'A2', 'A3', 'A4', 'V'])
        wr.writerows(zip(episode_buffer[:, 1], episode_buffer[:, 2], np.vstack(episode_buffer[:, 7])[:, 0],
                         np.vstack(episode_buffer[:, 7])[:, 1], np.vstack(episode_buffer[:, 7])[:, 2],
                         np.vstack(episode_buffer[:, 7])[:, 3], np.vstack(episode_buffer[:, 7])[:, 4],
                         episode_buffer[:, 8]))


# This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS) / duration * t)]
        except:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps=len(images) / duration, verbose=False)
        # clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps=len(images) / duration, verbose=False)
