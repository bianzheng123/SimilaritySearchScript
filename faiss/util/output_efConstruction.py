import json
import matplotlib.pyplot as plt
import matplotlib as mpl


def read_get_data(fname):
    file = open(fname, "r")
    json_data = json.load(file)
    x = list()
    y = list()

    for ele in json_data:
        x.append(ele['search_time_per_query'])
        y.append(ele['recall'])

    return x, y


ef_20 = read_get_data("../result/efConstruction/change_efConstruction_20.json")
ef_40 = read_get_data("../result/efConstruction/change_efConstruction_40.json")
ef_60 = read_get_data("../result/efConstruction/change_efConstruction_60.json")
ef_80 = read_get_data("../result/efConstruction/change_efConstruction_80.json")
ef_100 = read_get_data("../result/efConstruction/change_efConstruction_100.json")
ef_120 = read_get_data("../result/efConstruction/change_efConstruction_120.json")

# def smooth(x, y):
#     x_new = np.linspace(x.min(), x.max(), 300)  # 300 represents number of points to make between T.min and T.max
#     y_new = spline(x, y, x_new)
#     return x_new, y_new


# ef_20 = smooth(ef_20[0], ef_20[1])
# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
l_20, = plt.plot(ef_20[0], ef_20[1], marker='o', linestyle='solid', label='$efConstruction$: 20', color='#b9529f')
l_40, = plt.plot(ef_40[0], ef_40[1], marker='v', linestyle='solid', label='$efConstruction$: 40', color='#3953a4')
l_60, = plt.plot(ef_60[0], ef_60[1], marker='^', linestyle='solid', label='$efConstruction$: 60', color='#ed2024')
l_80, = plt.plot(ef_80[0], ef_80[1], marker='s', linestyle='solid', label='$efConstruction$: 80', color='#231f20')
l_100, = plt.plot(ef_100[0], ef_100[1], marker='*', linestyle='solid', label='$efConstruction$: 100', color='#098140')
l_120, = plt.plot(ef_120[0], ef_120[1], marker='X', linestyle='solid', label='$efConstruction$: 120', color='#7f8133')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.legend(loc='best', title="SIFT1M d=128,10-NN\n$M$=16")

plt.xlabel("Query time, ms")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
