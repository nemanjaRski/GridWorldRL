import numpy as np
import csv
import time
import os


# This is a simple function to reshape our game frames.
def process_state(state1):
    return np.reshape(state1, [21168])


def log_game(print_every, green, red, stuck, num_episode, rewards, prob_random):
    mean_green = np.mean(green[-print_every:])
    mean_red = np.mean(red[-print_every:])
    mean_stuck = np.mean(stuck[-print_every:])
    mean_reward = np.mean(rewards[-print_every:])
    current_time = time.strftime("%H:%M:%S", time.localtime())

    print(
        "Time: {} Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Green: {:0.04f} , Red: {:0.04f} , Stuck: {:0.04f}".format(
            current_time, num_episode, mean_reward, prob_random, mean_green, mean_red, mean_stuck))


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
def save_to_center(i, rList, jList, bufferArray, summaryLength, h_size, sess, mainQN, time_per_step, path,
                   save_full_state=False):
    if not os.path.exists(f'{path}/frames'):
        os.makedirs(f'{path}/frames')
    with open(f"{path}/log.csv", 'a') as myfile:

        images = list(bufferArray[:, 0])
        images.append(bufferArray[-1, 3])
        make_gif(np.array(images), f'{path}/frames/image' + str(i) + '.gif', duration=len(images) * time_per_step,
                 true_image=True, salience=False)
        if save_full_state:
            full_images = list(bufferArray[:, 5])
            make_gif(np.array(full_images), f'{path}/frames/full_image' + str(i) + '.gif',
                     duration=len(full_images) * time_per_step,
                     true_image=True, salience=False)

        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([i, np.mean(jList[-100:]), np.mean(rList[-summaryLength:]), './frames/image' + str(i) + '.gif',
                     './frames/log' + str(i) + '.csv', './frames/sal' + str(i) + '.gif'])
        myfile.close()
    with open(f'{path}/frames/log' + str(i) + '.csv', 'w') as myfile:
        state_train = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["ACTION", "REWARD", "A0", "A1", 'A2', 'A3', 'V'])
        a, v = sess.run([mainQN.advantage, mainQN.value],
                        feed_dict={mainQN.image_in: np.array([*bufferArray[:, 0]]),
                                   mainQN.train_length: len(bufferArray), mainQN.rnn_state_in: state_train,
                                   mainQN.batch_size: 1})
        wr.writerows(zip(bufferArray[:, 1], bufferArray[:, 2], a[:, 0], a[:, 1], a[:, 2], a[:, 3], v[:, 0]))


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
