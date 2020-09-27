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


data_size_1 = read_get_data("../result/data_size/change_data_size_1M.json")
data_size_10 = read_get_data("../result/data_size/change_data_size_10M.json")
data_size_20 = read_get_data("../result/data_size/change_data_size_20M.json")
data_size_50 = read_get_data("../result/data_size/change_data_size_50M.json")

# def smooth(x, y):
#     x_new = np.linspace(x.min(), x.max(), 300)  # 300 represents number of points to make between T.min and T.max
#     y_new = spline(x, y, x_new)
#     return x_new, y_new


# ef_20 = smooth(ef_20[0], ef_20[1])
# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
l_1, = plt.plot(data_size_1[0], data_size_1[1], marker='o', linestyle='solid', label='data size: 1M', color='#b9529f')
l_10, = plt.plot(data_size_10[0], data_size_10[1], marker='v', linestyle='solid', label='data size: 10M', color='#3953a4')
l_20, = plt.plot(data_size_20[0], data_size_20[1], marker='^', linestyle='solid', label='data size: 20M', color='#ed2024')
l_50, = plt.plot(data_size_50[0], data_size_50[1], marker='X', linestyle='solid', label='data size: 50M', color='#231f20')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.legend(loc='best', title="SIFT in different size\nd=128,10-NN\n$M$=40,$efConstruction$=100")

plt.xlabel("Query time, ms")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
