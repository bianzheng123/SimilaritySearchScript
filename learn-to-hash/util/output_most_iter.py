import json
import matplotlib.pyplot as plt
import numpy as np


def read_get_data(fname, n_candidates, n_bins):
    file = open(fname, "r")
    json_data = json.load(file)
    y = list()
    return n_candidates, len(json_data)


def get_diff_bin(start, end, n_bins):
    x_arr = []
    y_arr = []
    for number in range(start, end, 3):
        json_dir = '../result/%dK/sift_train.json' % (number)
        x, y = read_get_data(json_dir, number, n_bins)
        x_arr.append(x)
        y_arr.append(y)
    return x_arr, y_arr


curve = get_diff_bin(50, 250, 1)
# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
line1, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', color='#b9529f')
# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.legend(loc='lower left', title="SIFT, one level, 10-NN, 16bins")

plt.xlabel("different size")
plt.ylabel("bins to achieve 95%")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
