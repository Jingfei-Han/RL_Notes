#encoding:utf-8
import numpy as np

class Bandit(object):
    def __init__(self, k=10):
        self.k = k
        self.values = list(np.random.randn(self.k))
        self.optimal = np.argmax(self.values)

    def step(self, action):
        reward = np.random.randn() + self.values[action]
        return reward

    def get_optimal_action(self):
        return self.optimal

