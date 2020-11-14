import json
import matplotlib.pyplot as plt
import numpy as np


def get_cluster(json_dir):
    x_arr = []
    x_95_arr = []
    y_arr = []
    with open(json_dir, 'r') as file:
        json_data = json.load(file)
        for ele in json_data:
            if ele['acc'] == 0.0:
                continue
            x_arr.append(ele['probe'])
            x_95_arr.append(ele['probe95'])
            y_arr.append(ele['acc'])
    return x_arr, x_95_arr, y_arr

fname = '../result/sift/'
dir_cluster_16 = 'sift_cluster_16.json'
dir_cluster_256 = 'sift_cluster_256.json'

cls_16 = get_cluster(fname + dir_cluster_16)
cls_256 = get_cluster(fname + dir_cluster_256)
# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
line1, = plt.plot(cls_16[0], cls_16[2], marker='o', linestyle='solid', color='#fc0d1b', label='16 bin')
line2, = plt.plot(cls_16[1], cls_16[2], marker='v', linestyle='dotted', color='#fc0d1b', label='16 bin 95-quantile')

line1, = plt.plot(cls_256[0], cls_256[2], marker='^', linestyle='solid', color='#1072bd', label='256 bin')
line2, = plt.plot(cls_256[1], cls_256[2], marker='<', linestyle='dotted', color='#1072bd', label='256 bin 95-quantile')
# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.legend(loc='lower left', title="SIFT, one level, 10-NN, 16bins")

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
