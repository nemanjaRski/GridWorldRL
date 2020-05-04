import numpy as np
from scipy.special import softmax


def greedy_predict(q_values, epsilon=0):
    return np.argmax(q_values)


def epsilon_greedy_predict(q_values, epsilon=0):
    if np.random.rand(1) < epsilon:
        return np.random.randint(0, q_values.size)
    else:
        return greedy_predict(q_values, epsilon)


def random_predict(q_values, epsilon=0):
    return np.random.randint(0, q_values.size)


def boltzman_predict(q_values, epsilon=0):
    q_boltzman = softmax(q_values / epsilon)
    action_value = np.random.choice(q_boltzman[0], p=q_boltzman[0])
    return np.argmax(q_boltzman == action_value)
