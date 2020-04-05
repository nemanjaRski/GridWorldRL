import numpy as np
import tensorflow as tf

bandits = [0.2, 0, -0.2, -0.31, -0.3]
num_bandits = len(bandits)

def pull_bandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

total_episodes = 5000
total_reward = np.zeros(num_bandits)

e = 0.1

model = tf.keras.layers.Dense(1, input_shape=(num_bandits,), use_bias=False)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

def choose_action(model):
    if not model.weights:
        return np.random.randint(num_bandits)
    else:
        return np.argmax(model.weights)

def loss_function(responsible_weight, reward):
    return -(tf.math.log(responsible_weight)*reward)

def get_action_index(weights, action):
    index = 0
    for index in range(num_bandits):
        if weights[index] == action:
            return index
        index += 1
    return index



i = 0
with tf.GradientTape() as tape:
    while i < total_episodes:
        action_1_hot = np.expand_dims(tf.keras.utils.to_categorical(choose_action(model), num_bandits), axis=0)
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = model(action_1_hot).numpy()[0][0]
            action = get_action_index(model.weights[0], action)
        reward = pull_bandit(bandits[action])
        responsible_weight = model.trainable_weights[0][action].numpy()[0]
        total_reward[action] += reward
        loss_value = loss_function(responsible_weight=responsible_weight, reward=reward)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if i % 50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
        i += 1
# print(model.weights)
# print("The agent thinks bandit " + str(np.argmax(model.weights) + 1) + " is the most promising...")
# if np.argmax(model.weights) == np.argmax(-np.array(bandits)):
#     print("...and it was right!")
# else:
#     print("...and it was wrong!")