import numpy as np

LAST_POINT_ADJUST_TIME = 30000


class Adjust_env(object):

    def __init__(self):
        self.comp_dev = 0
        self.comp_std = 0
        self.last_comp_std = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0
        self.s = np.hstack([self.comp_dev, [self.comp_std, self.comp_proportion]])

    def reset(self):
        self.comp_dev = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0
        self.comp_std = 0
        self.last_comp_std = 0
        self.s = np.hstack([self.comp_dev, [self.comp_std, self.comp_proportion]])
        return self.s

    def step(self, action):
        self.comp_dev = self.comp_dev + action[0]
        if self.last_comp_proportion != 0 & self.last_comp_std != 0:
            reward = self.comp_proportion / self.last_comp_proportion - self.comp_std / self.last_comp_std
        else:
            reward = 0
        self.s = np.hstack([self.comp_dev, [self.comp_std, self.comp_proportion]])
        return self.s, reward

    def render(self):
        return self.s

    def update(self, update):
        self.last_comp_proportion = self.comp_proportion
        self.last_comp_std = self.comp_std
        self.comp_proportion = update['comp_proportion']
        self.comp_std = update['comp_std']
