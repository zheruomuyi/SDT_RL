import numpy as np

LAST_POINT_ADJUST_TIME = 30000


class Adjust_env(object):

    def __init__(self):
        self.comp_dev = 0
        self.comp_std = 0
        self.last_comp_std = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0
        self.comp_step = 0
        self.s = np.array([0, 0, 0])

    def reset(self):
        self.comp_dev = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0
        self.comp_std = 0
        self.last_comp_std = 0
        self.step_length = 0
        self.s = np.hstack([0, 0, 0])
        return self.s

    def step(self, action):
        comp_dev = min(self.comp_dev + (0.25 * action[0] * min(self.comp_dev, self.comp_step)), 10 * self.comp_step)
        comp_dev = max(comp_dev, 0)
        # comp_dev = self.comp_dev + 0.125 * action[0] * self.comp_step
        self.comp_dev = comp_dev
        reward = 0
        if self.last_comp_proportion < self.comp_proportion and self.last_comp_std > self.comp_std:
            reward = reward
        elif self.last_comp_proportion < self.comp_proportion and self.last_comp_std < self.comp_std:
            reward = reward - 0.2
        elif self.last_comp_proportion > self.comp_proportion and self.last_comp_std < self.comp_std:
            reward = reward - 1
        else:
            reward = reward - 0.5
        self.s = self.getstate()
        return self.s, reward

    def update(self, update):
        self.last_comp_proportion = self.comp_proportion
        self.last_comp_std = self.comp_std
        self.comp_proportion = update['comp_proportion']
        self.comp_std = update['comp_std']
        self.comp_dev = update['comp_dev']
        self.comp_step = update['comp_step']

    def getstate(self):
        proportion_ratio, std_ratio = 0, 0
        if self.comp_proportion != 0:
            proportion_ratio = self.last_comp_proportion / self.comp_proportion
        if self.comp_std != 0:
            std_ratio = self.last_comp_std / self.comp_std
        self.s = np.hstack([self.comp_dev, proportion_ratio, std_ratio])
        return self.s
