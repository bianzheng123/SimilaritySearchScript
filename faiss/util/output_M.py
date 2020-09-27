import json
import matplotlib.pyplot as plt


def read_get_data(fname):
    file = open(fname, "r")
    json_data = json.load(file)
    x = list()
    y = list()

    for ele in json_data:
        x.append(ele['search_time_per_query'])
        y.append(ele['recall'])

    return x, y


ef_2 = read_get_data("../result/M/change_M_2.json")
ef_3 = read_get_data("../result/M/change_M_3.json")
ef_6 = read_get_data("../result/M/change_M_6.json")
ef_12 = read_get_data("../result/M/change_M_12.json")
ef_20 = read_get_data("../result/M/change_M_20.json")
ef_40 = read_get_data("../result/M/change_M_40.json")
ef_60 = read_get_data("../result/M/change_M_60.json")

# def smooth(x, y):
#     x_new = np.linspace(x.min(), x.max(), 300)  # 300 represents number of points to make between T.min and T.max
#     y_new = spline(x, y, x_new)
#     return x_new, y_new


# ef_20 = smooth(ef_20[0], ef_20[1])
# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
l_2, = plt.plot(ef_2[0], ef_2[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')
l_3, = plt.plot(ef_3[0], ef_3[1], marker='v', linestyle='solid', label='$M$: 3', color='#3953a4')
l_6, = plt.plot(ef_6[0], ef_6[1], marker='^', linestyle='solid', label='$M$: 6', color='#ed2024')
l_12, = plt.plot(ef_12[0], ef_12[1], marker='<', linestyle='solid', label='$M$: 12', color='#231f20')
l_20, = plt.plot(ef_20[0], ef_20[1], marker='>', linestyle='solid', label='$M$: 20', color='#098140')
l_40, = plt.plot(ef_40[0], ef_40[1], marker='s', linestyle='solid', label='$M$: 40', color='#7f8133')
l_60, = plt.plot(ef_60[0], ef_60[1], marker='X', linestyle='solid', label='$M$: 60', color='#d28bbc')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.legend(loc='best', title="SIFT1M d=128,10-NN\n$efConstruction$=100")

plt.xlabel("Query time, ms")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
