import numpy as np

LAST_POINT_ADJUST_TIME = 30000


class Adjust_env(object):

    def __init__(self):
        self.comp_dev = 0
        self.comp_std = 0
        self.comp_proportion = 0
        self.s = np.array([self.comp_std, self.comp_dev])

    def reset(self):
        self.comp_dev = 0
        self.comp_proportion = 0
        self.comp_std = 0
        self.s = np.array([self.comp_std, self.comp_dev])
        return self.s

    def step(self, action):
        self.comp_dev = self.comp_dev + action["change"]
        reward = action['comp_proportion'] / self.comp_proportion - action['comp_std'] / self.comp_std
        self.comp_std = action['comp_std']
        self.comp_proportion = action['comp_proportion']
        self.s = {'comp_dev': self.comp_dev, 'comp_std': self.comp_std}
        done = 0
        return self.comp_dev, reward, done

    def render(self):
        return self.s, self.comp_std
