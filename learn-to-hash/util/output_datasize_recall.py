import json
import matplotlib.pyplot as plt
import numpy as np


def read_get_data(fname):
    file = open(fname, "r")
    json_data = json.load(file)
    x = list()
    y = list()

    for ele in json_data:
        x.append(ele['dataset_size'])
        y.append(ele['recall'])

    return x, y


curve = read_get_data("../result/single_16/single_16.json")

# def smooth(x, y):
#     x_new = np.linspace(x.min(), x.max(), 300)  # 300 represents number of points to make between T.min and T.max
#     y_new = spline(x, y, x_new)
#     return x_new, y_new


# curve = smooth(curve[0], curve[1])
# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', color='#b9529f')
# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.legend(loc='lower left', title="SIFT, one level, 10-NN,$S$=5, 16bins")

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
