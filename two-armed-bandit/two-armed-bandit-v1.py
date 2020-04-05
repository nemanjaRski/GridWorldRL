import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

bandits = [0.2, 0, -0.2, -0.31, -0.3]
num_bandits = len(bandits)

def pull_bandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

tf.reset_default_graph()

weights = tf.Variable(tf.ones([num_bandits])) # [[1,1,1,1]]

chosen_action = tf.argmax(weights, 0)

reward_holder = tf.placeholder(shape=[1], dtype=tf.float32) # [1] || [-1]
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

total_episodes = 5000
total_reward = np.zeros(num_bandits)

e = 1

init = tf.initialize_all_variables()

action_array = [0,0,0,0,0]
action_array_rewards = [0,0,0,0,0]
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pull_bandit(bandits[action])
        action_array[action] += 1
        total_reward[action] += reward
        action_array_rewards[action] += (reward > 0)
        _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={reward_holder:[total_reward[action]/action_array[action]], action_holder:[action]})
        if i % 50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
        i+=1
print("The agent thinks bandit " + str(np.argmax(ww) + 1) + " is the most promising...")
print("Number of pulls: " + str(action_array))
print("Number of good rewards: " + str(action_array_rewards))
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")


