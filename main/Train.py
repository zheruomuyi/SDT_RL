import numpy as np
from environment import Adjust_env
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

SAVE_PATH = "../model/rl_model"
DATA_PATH = "../data/"
MIN_BATCH_SIZE = 64
buffer_s, buffer_a, buffer_r = [], [], []
t = 0
GAMMA = 0.99  # reward discount factor
GLOBAL_RUNNING_R = []


def calculate(abandon, compress_info):
    last_value = compress_info['last_value']
    last_time = compress_info['last_time']
    last_stored_value = compress_info['last_stored_value']
    last_stored_time = compress_info['last_stored_time']
    error = 0
    diff = last_value - last_stored_value
    for kv in abandon:
        t = kv[0]
        v = kv[1]
        error += pow(v - last_value + diff * (t - last_time) / (last_stored_time - last_time), 2)
    compress_info['error'] += error
    compress_info['comp_std'] = compress_info['error'] / (compress_info['before_comp'] - compress_info['after_comp'])


class Train(object):
    def __init__(self, ppo):
        self.ppo = ppo

    def learn(self):
        for filenames in os.walk(DATA_PATH):
            for file in filenames:
                if file.endswith(".csv"):
                    print(file)
                    data = pd.read_csv(DATA_PATH + file)
                    for key in data:
                        time_stamp = data['timeOfDay']
                        values = data[key]
                        self.compress(time_stamp, values)

        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.xlabel("adjust time")
        plt.ylabel('reward')
        plt.savefig('../image/train.jpg')
        plt.show()

    def compress(self, time_stamp, values):
        compress_info = {'up_gate': - sys.float_info.max, 'down_gate': sys.float_info.max, 'comp_dev': 0
            , 'first_adjust': 0, 'before_comp': 0, 'after_comp': 0, 'error': 0}
        after_compress = []
        abandon = []
        first_collector = []
        env = Adjust_env()
        for index, value in time_stamp:
            value = values[index]
            if pd.isnull(value):
                continue
            compress_info['before_comp'] += 1
            after_compress.append([index, value])
            if len(first_collector) < 100 and compress_info['first_adjust'] == 0:
                first_collector.append([index, value])
                after_compress.append([index, value])
                compress_info['after_comp'] += 1
                compress_info['last_value'] = values[index]
                compress_info['last_time'] = index
                continue
            elif compress_info['first_adjust'] == 0 and len(first_collector) == 100:
                compress_info['comp_dev'] = (np.std(np.array(first_collector))) / 2
                compress_info['comp_step'] = compress_info['comp_dev'] / 3
                compress_info['next_time'] = index + 300
                compress_info['last_stored_value'] = values[index]
                compress_info['last_stored_time'] = index
                compress_info['last_value'] = values[index]
                compress_info['last_time'] = index
                compress_info['after_comp'] += 1
                print('init compDev :', compress_info['comp_dev'])
                continue

            ts = index - compress_info['last_stored_time']
            if ts != 0:
                now_gate = (value - compress_info['last_stored_value']) / ts
                if now_gate < compress_info['up_gate'] or now_gate > compress_info['down_gate']:
                    compress_info['last_stored_value'] = values[index]
                    compress_info['last_stored_time'] = index
                    after_compress.append([index, value])
                    compress_info['after_comp'] += 1
                    calculate(abandon, compress_info)

                    if index >= compress_info['next_time']:
                        compress_info['comp_proportion'] = compress_info['before_comp'] / compress_info['after_comp']
                        old_dev = compress_info['comp_dev']
                        self.adjust_para(compress_info, env)
                        print('adjust comp_dev :', old_dev, ' to ', compress_info['comp_dev'],
                              'comp_proportion :', compress_info['comp_proportion'], + "comp_std :",
                              compress_info['comp_std'])
                        compress_info['next_time'] += 300
                else:
                    abandon.append([index, value])
                    now_up = (value - compress_info['last_stored_value'] - compress_info['comp_dev']) / ts
                    now_down = (value - compress_info['last_stored_value'] + compress_info['comp_dev']) / ts
                    compress_info['up_gate'] = max(now_up, compress_info['up_gate'])
                    compress_info['down_gate'] = min(now_down, compress_info['down_gate'])
            compress_info['last_value'] = values[index]
            compress_info['last_time'] = index

    def adjust_para(self, compress_info, env):
        global buffer_s, buffer_a, buffer_r, ep_r, t
        comp_dev_old = compress_info['comp_dev']
        comp_std = compress_info['comp_std']
        comp_proportion = compress_info['comp_proportion']
        comp_step = compress_info['comp_step']
        update = {'comp_dev': comp_dev_old, 'comp_proportion': comp_proportion, 'comp_std': comp_std,
                  'comp_step': comp_step}
        env.update(update)
        s = env.getstate()
        a = self.ppo.choose_action(s)
        s_, r = env.step(a)
        comp_dev = s_[0]
        print('update', ' compDev from ', comp_dev_old, ' to ', comp_dev)
        compress_info[comp_std] = comp_dev
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)

        ep_r += r
        t += 1
        if t >= MIN_BATCH_SIZE:
            v_s_ = self.ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            self.ppo.update(bs, ba, br)
            self.ppo.SAVER.save(self.ppo.sess, SAVE_PATH)
            buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer
        if len(t) == 0:
            GLOBAL_RUNNING_R.append(ep_r)
        else:
            GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
        print(t, '|Ep_r' + ': %.5f' % ep_r, )