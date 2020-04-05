import numpy as np
import matplotlib.pyplot as plt
import gym
import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import glorot_uniform

env = gym.make("CartPole-v0")


def get_policy_model(env, hidden_layer_neurons, lr):
    dimen = env.reset().shape
    num_actions = env.action_space.n

    input = layers.Input(shape=dimen, name="input_x")
    advantage = layers.Input(shape=[1], name="advantages")

    hidden_layer = layers.Dense(hidden_layer_neurons,
                                activation="relu",
                                use_bias=False,
                                kernel_initializer=glorot_uniform(seed=42),
                                name="hidden_layer")(input)
    output = layers.Dense(num_actions,
                          activation="softmax",
                          kernel_initializer=glorot_uniform(seed=42),
                          use_bias=False,
                          name="output")(hidden_layer)

    def custom_loss(y_true, y_pred):
        log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
        return K.mean(log_lik * advantage, keepdims=True)

    model_train = Model(inputs=[input, advantage], outputs=output)
    model_train.compile(loss=custom_loss, optimizer=Adam(lr))
    model_predict = Model(inputs=[input], outputs=output)
    return model_train, model_predict


def discount_rewards(r, gamma=0.99):
    prior = 0
    out = []
    for val in r:
        new_val = val + prior * gamma
        out.append(new_val)
        prior = new_val
    return np.array(out[::-1])


hidden_layer_neurons = 8
gamma = .99
dimen = len(env.reset())
print_every = 100
batch_size = 5
num_episodes = 10000
render = False
lr = 1e-2
goal = 190


def score_model(model, num_tests, render=False):
    scores = []
    for num_test in range(num_tests):
        observation = env.reset()
        reward_sum = 0
        while True:
            if render:
                env.render()

            state = np.reshape(observation, [1, dimen])
            predict = model.predict([state])[0]
            action = np.argmax(predict)
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
    env.close()
    return np.mean(scores)


model_train, model_predict = get_policy_model(env, hidden_layer_neurons, lr)
model_predict.summary()

reward_sum = 0
num_actions = env.action_space.n

states = np.empty(0).reshape(0, dimen)
actions = np.empty(0).reshape(0, 1)
rewards = np.empty(0).reshape(0, 1)
discounted_rewards = np.empty(0).reshape(0, 1)

observation = env.reset()

num_episode = 0

losses = []

while num_episode < num_episodes:
    state = np.reshape(observation, [1, dimen])

    predict = model_predict.predict([state])[0]
    action = np.random.choice(range(num_actions), p=predict)

    states = np.vstack([states, state])
    actions = np.vstack([actions, action])

    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    rewards = np.vstack([rewards, reward])

    if done:
        discounted_rewards_episode = discount_rewards(rewards, gamma)
        discounted_rewards = np.vstack([discounted_rewards, discounted_rewards_episode])

        rewards = np.empty(0).reshape(0, 1)

        if (num_episode + 1) % batch_size == 0:
            discounted_rewards -= discounted_rewards.mean()
            discounted_rewards /= discounted_rewards.std()
            discounted_rewards = discounted_rewards.squeeze()
            actions = actions.squeeze().astype(int)

            actions_train = np.zeros([len(actions), num_actions])
            actions_train[np.arange(len(actions)), actions] = 1

            loss = model_train.train_on_batch([states, discounted_rewards], actions_train)
            losses.append(loss)

            states = np.empty(0).reshape(0, dimen)
            actions = np.empty(0).reshape(0, 1)
            discounted_rewards = np.empty(0).reshape(0, 1)

        if (num_episode + 1) % print_every == 0:
            score = score_model(model_predict, 10)
            print("Average reward for training episode {}: {:0.2f} Test Score: {:0.2f} Loss: {:0.6f} ".format(
                (num_episode + 1), reward_sum/print_every,
                score,
                np.mean(losses[-print_every:])
            ))
            if score >= goal:
                print("Solved in {} episodes!".format(num_episodes))
                break
            reward_sum = 0;
        num_episode += 1
        observation = env.reset()

score = score_model(model_predict, 10, True)