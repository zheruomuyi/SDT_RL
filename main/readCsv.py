import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import sys
import csv

DATA_PATH = "../data/"
for (dirpath, dirnames, filenames) in os.walk(DATA_PATH):
    for file in filenames:
        if file.endswith(".csv"):
            print(file)
            data = pd.read_csv(DATA_PATH + file)
            for key in data:
                time_stamp = data['timeOfDay']
                values = data[key]

# data = pd.read_csv('LiJ4HoBd@compress@double1.csv')  # 导入csv文件
#
# y = data['double'].T.values  # 设置y轴数值 ,.T是转置
#
# x = data['time'].T.values
# plt.figure(figsize=(10, 6))
#
# # plt.plot(x, y, '')
#
# data1 = pd.read_csv('test_local_double1_end.csv')  # 导入csv文件
#
# y1 = data1['double'].T.values  # 设置y轴数值 ,.T是转置
#
# x1 = data1['time'].T.values
#
# # plt.plot(x1, y1, '')
#
# data2 = pd.read_csv('end rl.csv')  # 导入csv文件
#
# y2 = data2['double'].T.values  # 设置y轴数值 ,.T是转置
#
# x2 = data2['time'].T.values
#
# # plt.plot(x2, y2, '')
#
# fig = plt.figure(dpi=128, figsize=(10, 8))
# plt.plot(x, y, c='red', alpha=0.5)  # 实参alpha指定颜色的透明度，0表示完全透明，1（默认值）完全不透明
# # plt.plot(x1, y1, c='blue', alpha=0.5)
# # plt.plot(x2, y2, c='blue', alpha=0.5)
# # plt.fill_between(x, y, x1, y1, facecolor='blue', alpha=0.1)  # 给图表区域填充颜色
#
# plt.xlabel('timestamp')
#
# plt.ylabel('number')
#
# plt.show()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# t = np.arange(0.0, 1.0, 0.0001)
# s = np.sin(6 * np.pi * t)
# plt.rcParams['lines.color'] = 'r'
# plt.plot(t, s)
# plt.show()
# # # 任意的多组列表
# # a = [1, 2, 3]
# # b = [4, 5, 6]
# #
# # # 字典中的key值即为csv中列名
# dataframe = pd.DataFrame({'time': t, 'double': s})
# #
# # # 将DataFrame存储为csv,index表示是否显示行名，default=True
# dataframe.to_csv("test.csv", index=False, sep=',')
