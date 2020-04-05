from __future__ import print_function
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym

env = gym.make('CartPole-v0')

H = 8
learning_rate = 1e-2
gamma = 0.99
decay_rate = 0.99
resume = False

model_bs = 3
real_bs = 3

D = 4

tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None, 4], name="input_x")
W1 = tf.get_variable("W1", shape=[4, H], initializer=tf.initializers.glorot_uniform())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.initializers.glorot_uniform())

score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1_grad = tf.placeholder(tf.float32, name="batch_grad1")
W2_grad = tf.placeholder(tf.float32, name="batch_grad2")
batch_grad = [W1_grad, W2_grad]
loglik = tf.log(input_y*(input_y - probability) + (1-input_y)*(input_y+probability))
loss = -tf.reduce_mean(loglik * advantages)
new_grads = tf.gradients(loss, tvars)
update_grads = adam.apply_gradients(zip(batch_grad, tvars))

mH = 256

input_data = tf.placeholder(tf.float32, [None, 5])
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [mH, 50])
    softmax_b = tf.get_variable("softmax_b", [50])

previous_state = tf.placeholder(tf.float32, [None, 5], name="previous_state")
W1M = tf.get_variable("W1M", shape=[5, mH], initializer=tf.initializers.glorot_uniform())
B1M = tf.Variable(tf.zeros([mH]), name="B1M")
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)
W2M = tf.get_variable("W2M", shape=[mH, mH], initializer=tf.initializers.glorot_uniform())
B2M = tf.Variable(tf.zeros([mH]), name="B2M")
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)

wO = tf.get_variable("wO", shape=[mH, 4], initializer=tf.initializers.glorot_uniform())
wR = tf.get_variable("wR", shape=[mH, 1], initializer=tf.initializers.glorot_uniform())
wD = tf.get_variable("wD", shape=[mH, 1], initializer=tf.initializers.glorot_uniform())

bO = tf.Variable(tf.zeros([4]), name="bO")
bR = tf.Variable(tf.zeros([1]), name="bR")
bD = tf.Variable(tf.zeros([1]), name="bD")

predicted_observations = tf.matmul(layer2M, wO, name="predicted_observations") + bO
predicted_reward = tf.matmul(layer2M, wR, name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name="predicted_done") + bD)

true_observation = tf.placeholder(tf.float32, [None, 4], name="true_observation")
true_reward = tf.placeholder(tf.float32, [None, 1], name="true_reward")
true_done = tf.placeholder(tf.float32, [None, 1], name="true_done")

predicted_state = tf.concat([predicted_observations, predicted_reward, predicted_done], 1)

observation_loss = tf.square(true_observation - predicted_observations)

reward_loss = tf.square(true_reward - predicted_reward)

done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done)
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

model_adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_model = model_adam.minimize(model_loss)


def reset_grad_buffer(grad_buffer):
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad * 0
    return grad_buffer


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def step_model(sess, xs, action):
    to_feed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1,5])
    my_predict = sess.run([predicted_state], feed_dict={previous_state: to_feed})
    reward = my_predict[0][:, 4]
    observation = my_predict[0][:, 0:4]
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = np.clip(my_predict[0][:, 5], 0, 1)
    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done

xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs

draw_from_model = False
train_the_model = True
train_the_policy = False

switch_point = 1

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    x = observation
    grad_buffer = sess.run(tvars)
    grad_buffer = reset_grad_buffer(grad_buffer)

    while episode_number <= 5000:
        if (reward_sum/batch_size > 150 and draw_from_model == False) or rendering == True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1,4])
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        if draw_from_model == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = step_model(sess, xs, action)

        reward_sum += reward

        ds.append(done*1)
        drs.append(reward)

        if done:
            if draw_from_model == False:
                real_episodes += 1
            episode_number += 1

            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)

            xs, drs, ys, ds = [], [], [], []

            if train_the_model == True:
                actions = np.array([np.abs(y-1) for y in epy][:-1])
                state_prevs = epx[:-1, :]
                state_prevs = np.hstack([state_prevs, actions])
                state_nexts = epx[1:, :]
                rewards = np.array(epr[1:, :])
                dones = np.array(epd[1:, :])
                state_nexts_all = np.hstack([state_nexts, rewards, dones])

                feed_dict = {previous_state: state_prevs, true_observation: state_nexts, true_done: dones, true_reward: rewards}
                loss, p_state, _ = sess.run([model_loss, predicted_state, update_model], feed_dict)

            if train_the_policy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                t_grad = sess.run(new_grads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

                if np.sum(t_grad[0] == t_grad[0]) == 0:
                    break
                for ix, grad in enumerate(t_grad):
                    grad_buffer[ix] += grad

            if switch_point + batch_size == episode_number:
                switch_point = episode_number
                if train_the_policy == True:
                    sess.run(update_grads, feed_dict={W1_grad: grad_buffer[0], W2_grad: grad_buffer[1]})
                    grad_buffer = reset_grad_buffer(grad_buffer)

                running_reward = reward_sum if running_reward is None else running_reward * gamma + reward_sum * 0.01

                if draw_from_model == False:
                    print('World Perf: Episode %f. Reward %f. action %f. mean reward %f.' % (real_episodes, reward_sum/real_bs, action, running_reward/real_bs))
                    if reward_sum/batch_size > 200:
                        break
                reward_sum = 0

                if episode_number > 100:
                    draw_from_model = not draw_from_model
                    train_the_model = not train_the_model
                    train_the_policy = not train_the_policy
            if draw_from_model == True:
                observation = np.random.uniform(-0.1, 0.1, [4])
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = real_bs


print(real_episodes)