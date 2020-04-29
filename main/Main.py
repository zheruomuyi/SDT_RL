import tensorflow as tf
import tensorflow.compat.v1 as tfv
import numpy as np
import queue
from environment import Adjust_env
import matplotlib.pyplot as plt
from flask import Flask
from flask import jsonify
from flask import request
import threading
import os

EP_LEN = 1700
GAMMA = 0.99  # reward discount factor
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
UPDATE_STEP = 10  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective
S_DIM, A_DIM = 3, 1  # state and action dimension

SAVE_PATH = "../model/rl_model"
environments = {}
t = {}
ep_r = {}
buffer_s, buffer_a, buffer_r = [], [], []
tfv.disable_v2_behavior()
app = Flask(__name__)


@app.route('/adjust', methods=['POST'])
def adjust():
    if request.method == 'POST':
        try:
            last_compress_point = request.json
            key = last_compress_point['key']
            print(key)
            comp_dev = adjust_param(last_compress_point, key)
            return jsonify({'status': 0, "msg": 'OK', "data": comp_dev})
        except Exception as e:
            return jsonify({'status': 1, "msg": 'rl adjust had something wrong!' + e.__str__()})
    else:
        return jsonify({'status': 1, "msg": 'rl adjust api should be post'})


@app.route('/adjust', methods=['GET'])
def adjust_get():
    return jsonify({'status': 0, "msg": 'OK'})


class PPO(object):
    def __init__(self):
        self.sess = tfv.Session()
        self.tfs = tfv.placeholder(tfv.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope("critic"):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tfv.placeholder(tfv.float32, [None, 1], 'reward')
        self.advantage = self.tfdc_r - self.v
        self.closs = tfv.reduce_mean(tfv.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        actor, actor_params = self._build_anet('actor', trainable=True)
        old_actor, old_actor_params = self._build_anet('old_actor', trainable=False)
        self.sample_op = tfv.squeeze(actor.sample(1), axis=0)  # operation of choosing action
        self.update_old_actor_op = [oldp.assign(p) for p, oldp in zip(actor_params, old_actor_params)]

        self.tfa = tfv.placeholder(tfv.float32, [None, A_DIM], 'action')
        self.tfadv = tfv.placeholder(tfv.float32, [None, 1], 'advantage')
        ratio = tfv.exp(actor.log_prob(self.tfa) - old_actor.log_prob(self.tfa))
        # ratio = actor.prob(self.tfa) / (old_actor.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tfv.reduce_mean(tfv.minimum(  # clipped surrogate objective
            surr,
            tfv.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tfv.global_variables_initializer())

        self.writer = tf.summary.FileWriter("../logs/", self.sess.graph)

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            UPDATE_EVENT.wait()  # wait until get batch of data
            self.sess.run(self.update_old_actor_op)  # copy actor to old actor
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
            data = np.vstack(data)
            s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
            adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
            # update actor and critic in a update loop
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
            [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
            UPDATE_EVENT.clear()  # updating finished
            GLOBAL_UPDATE_COUNTER = 0  # reset counter
            ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, name, trainable):
        with tfv.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tfv.get_collection(tfv.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


def work(key, s, s_, a, r):
    global GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, buffer_s, buffer_a, buffer_r, ep_r, t
    if key not in ep_r:
        ep_r[key] = 0
    buffer_s.append(s)
    buffer_a.append(a)
    buffer_r.append(r)  # normalize reward, find to be useful
    if not ROLLING_EVENT.is_set():  # while global PPO is updating
        ROLLING_EVENT.wait()  # wait until PPO is updated
        buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer

    ep_r[key] += r
    GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size
    t[key] += 1
    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
        v_s_ = GLOBAL_PPO.get_v(s_)
        discounted_r = []  # compute discounted reward
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()

        bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
        buffer_s, buffer_a, buffer_r = [], [], []
        QUEUE.put(np.hstack((bs, ba, br)))
        if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
            ROLLING_EVENT.clear()  # stop collecting data
            UPDATE_EVENT.set()  # globalPPO update
            SAVER.save(GLOBAL_PPO.sess, SAVE_PATH)
            GLOBAL_UPDATE_COUNTER = 0

    # record reward changes, plot later
    if len(GLOBAL_RUNNING_R[key]) == 0:
        GLOBAL_RUNNING_R[key].append(ep_r[key])
    else:
        GLOBAL_RUNNING_R[key].append(GLOBAL_RUNNING_R[key][-1] * 0.9 + ep_r[key] * 0.1)
    print('{0:.1f}%'.format(t[key] / EP_LEN * 100), '|Ep_r' + key + ': %.2f' % ep_r[key], )


def adjust_param(last_compress_point, key):
    if key in environments and key in GLOBAL_RUNNING_R and key in t:
        env = environments[key]
    else:
        env = Adjust_env()
        environments[key] = env
        GLOBAL_RUNNING_R[key] = []
        t[key] = 1
    comp_dev_old = last_compress_point['comp_dev']
    comp_std = last_compress_point['comp_std']
    comp_proportion = last_compress_point['comp_proportion']
    comp_step = last_compress_point['comp_step']
    update = {'comp_dev': comp_dev_old, 'comp_proportion': comp_proportion, 'comp_std': comp_std,
              'comp_step': comp_step}
    env.update(update)
    s = env.getstate()
    a = GLOBAL_PPO.choose_action(s)
    s_, r = env.step(a)
    comp_dev = s_[0]
    thead_one = threading.Thread(target=work, args=(key, s, s_, a, r))
    thead_one.start()  # 准备就绪,等待cpu执行
    if t[key] == EP_LEN:
        plt.plot(np.arange(len(GLOBAL_RUNNING_R[key])), GLOBAL_RUNNING_R[key])
        plt.xlabel(key)
        plt.ylabel('reward')
        plt.savefig('../image/' + key + '.jpg')
        plt.show()

    print('update', ' compDev from ', comp_dev_old, ' to ', comp_dev)
    return comp_dev


if __name__ == '__main__':

    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    GLOBAL_PPO = PPO()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    COORD = tf.train.Coordinator()
    SAVER = tf.train.Saver()
    if os.path.isfile(SAVE_PATH + '.meta'):
        SAVER.restore(GLOBAL_PPO.sess, SAVE_PATH)
    GLOBAL_UPDATE_COUNTER = 0
    GLOBAL_RUNNING_R = {}
    QUEUE = queue.Queue()  # workers putting data in this queue

    thread = threading.Thread(target=GLOBAL_PPO.update, )
    thread.start()

    app.run(host='0.0.0.0', port=8080, debug=True)
    adjust_get.run(host='0.0.0.0', port=8080, debug=True)
    COORD.join(thread)
