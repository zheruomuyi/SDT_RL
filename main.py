from environment import Env
from RL_brain import DeepQNetwork
import requests
import json
import pymysql as mysql
import sys
import numpy as np
import pandas as pd

url = "http://tsdb-queryengine.beta-k8s-cn4.eniot.io/dataService/assets/tsdb/rawData/timeRange"


def run_main():
    step = 0
    for episode in range(300):
        observation = Env.reset()
        # DQN 根据观测值选择行为
        action = RL.choose_action(observation)

        # 环境根据行为给出下一个 state, reward, 是否终止
        observation_, reward, done = env.step(action)

        # DQN 存储记忆
        RL.store_transition(observation, action, reward, observation_)

        # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # 将下一个 state_ 变为 下次循环的 state
        observation = observation_

        # 如果终止, 就跳出循环
        if done:
            break
        step += 1  # 总步数

        # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    conn = mysql.connect(host="172.20.16.64", port=3306, user="root", password="root", database="compress_policy",
                         charset="utf8")
    try:
        with conn.cursor() as cursor:
            sql = "select * from model_point_storage_policy_info where `point_compress` = 1"
            cursor.execute(sql)
            result = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    # print(result)
    for item in result:
        # print(item)
        point_id = item[0]
        model_id = item[1]
        org_id = item[2]
        storage_policy_id = item[3]
        point_com_dev = json.loads(item[5])
        asset_ids = point_com_dev['assetIds']
        comp_dev = point_com_dev['compDev']
        print(org_id, '\t', model_id, '\t', asset_ids, '\t', storage_policy_id, '\t', comp_dev, '\t', point_id)

        querystring = {"accessKey": "EEOP_TEST", "orgId": org_id, "assetIds": 'Qaae2jze',
                       "measurepoints": point_id, 'pageSize': 640000,
                       "startTime": "2020-01-05 00:00:00", "endTime": "2020-01-05 23:59:59",
                       "localTimeAccuracy": "False"}
        response = requests.request("GET", url, params=querystring)
        asset_ids = ['Qaae2jze']
        data = json.loads(response.text)['data']['items']
        ts = pd.DataFrame(data)
        # ts['timestamp'] = pd.to_datetime(ts['timestamp'])
        print(len(ts))
        # for value_item in ts:
        #     timestamp = value_item['timestamp']
        #     value = 'null'
        #     if point_id in value_item:
        #         value = value_item[point_id]
        #     print(timestamp, '\t', value)
        env = Env(ts['timestamp',point_id])
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=2000,  # 记忆上限
                      # output_graph=True   # 是否输出 tensorboard 文件
                      )
    env.after(100, run_main)
    env.mainloop()
    RL.plot_cost()  # 观看神经网络的误差曲线
