import numpy as np
import tensorflow as tf
import tf_slim as slim

class contextual_bandit():
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_bandit(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pull_arm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1


class agent():
    def __init__(self, lr, s_size, a_size):
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)

        # output = slim.fully_connected(state_in_OH, a_size, biases_initializer=None,\
        #                               activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())

        weights = tf.Variable(tf.ones([s_size, a_size]))
        output = tf.matmul(state_in_OH, weights)
        self.output = tf.reshape(output, [-1]) #[[]] -> []

        self.chosen_action = tf.argmax(self.output, 0)

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)


tf.reset_default_graph()
c_bandit = contextual_bandit()
my_agent = agent(lr=0.001, s_size=c_bandit.num_bandits, a_size=c_bandit.num_actions)

weights = tf.trainable_variables()[0]

total_episodes = 10000
total_reward = np.zeros([c_bandit.num_bandits, c_bandit.num_actions])
e = 0.1

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        s = c_bandit.get_bandit()
        if np.random.rand(1) < e:
            action = np.random.randint(c_bandit.num_actions)
        else:
            action = sess.run(my_agent.chosen_action, feed_dict={my_agent.state_in:[s]})
        reward = c_bandit.pull_arm(action)
        feed_dict = {my_agent.reward_holder:[reward], my_agent.action_holder:[action], my_agent.state_in:[s]}
        _, ww = sess.run([my_agent.update, weights], feed_dict=feed_dict)
        total_reward[s, action] += reward
        if i % 500 == 0:
            print("Mean reward for each of the " + str(c_bandit.num_bandits) + " bandits: " + str(np.mean(total_reward, axis=1)))
        i += 1
for a in range(c_bandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising...")
    if np.argmax(ww[a]) == np.argmin(c_bandit.bandits[a]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")