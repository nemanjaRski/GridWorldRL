import gym
import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

env = gym.make('FrozenLake-v0', is_slippery = True)

tf.reset_default_graph()

inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))

Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 10000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
  sess.run(init)
  for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    if i % 100 == 0:
      print("Epoch: ", str(i), str(sum(rList) / (i + 1)) + "%")
    # The Q-Network
    while j < 99:
      j += 1
      # Choose an action by greedily (with e chance of random action) from the Q-network
      a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s + 1]})
      if np.random.rand(1) < e:
        a[0] = env.action_space.sample()
      # Get new state and reward from environment
      s1, r, d, _ = env.step(a[0])
      # Obtain the Q' values by feeding the new state through our network
      Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})
      # Obtain maxQ' and set our target value for chosen action.
      maxQ1 = np.max(Q1)
      targetQ = allQ
      targetQ[0, a[0]] = r + y * maxQ1
      # Train our network using target and predicted Q values
      _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
      rAll += r
      s = s1
      if d == True:
        break
    jList.append(j)
    rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
