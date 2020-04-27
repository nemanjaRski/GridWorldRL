import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
env = gym.make('FrozenLake-v0', is_slippery=False)

y = .99
exploration = 0.1
num_epochs = 4000
# try wihtout bias

model = tf.keras.layers.Dense(4, input_shape=(16,), use_bias=False)

mseLoss = tf.keras.losses.MeanSquaredError()
# try Adam as well
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

jList = []
rList = []

for epoch in range(num_epochs):
    print("Epoch: ", str(epoch))
    state = env.reset()
    reward_all = 0
    done = False
    max_steps = 0
    action = 0
    while max_steps < 99:
        max_steps += 1
        with tf.GradientTape() as tape:
            state_1_hot = np.expand_dims(tf.keras.utils.to_categorical(state, 16), axis=0)
            q_value = model(state_1_hot)
            action = np.argmax(q_value.numpy(), axis=1)[0]
            if np.random.rand(1) < exploration:
                action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            new_state_1_hot = np.expand_dims(tf.keras.utils.to_categorical(new_state, 16), axis=0)
            new_q_value = model(new_state_1_hot)

            max_new_q_value = np.max(new_q_value.numpy())
            target_q_value = q_value.numpy()
            target_q_value[0, action] = reward + y * max_new_q_value
            target_q_value = tf.convert_to_tensor(target_q_value)

            loss = mseLoss(target_q_value, q_value)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        reward_all += reward
        state = new_state

        if done:
            exploration = 1. / ((epoch / 50) + 10)
            # if exploration < 0.1:
            #     exploration = 0.1
            break
    jList.append(max_steps)
    rList.append(reward_all)

print("Percent of succesful episodes: " + str(sum(rList) * 100 / num_epochs) + "%")
print(model.weights)
plt.plot(rList)
plt.show()