import json
import matplotlib.pyplot as plt


def get_cluster_other(json_dir):
    x_arr = []
    y_arr = []
    with open(json_dir, 'r') as file:
        json_data = json.load(file)
        for ele in json_data:
            # if ele['recall'] == 0.0:
            #     continue
            if ele['acc'] == 0.0:
                continue
            # x_arr.append(ele['n_candidates_avg'])
            # y_arr.append(ele['recall'])
            x_arr.append(ele['probe'])
            y_arr.append(ele['acc'])
    return x_arr, y_arr


def get_cluster_nn_classification(json_dir):
    x_arr = []
    y_arr = []
    with open(json_dir, 'r') as file:
        json_data = json.load(file)
        for ele in json_data:
            if ele['recall'] == 0.0:
                continue
            x_arr.append(ele['n_candidate'])
            y_arr.append(ele['recall'])
    return x_arr, y_arr


# other_fname = '../siftsmall/learn-to-hash/'
# dir_learn_on_graph = other_fname + 'sift_cluster_16.json'

nn_classification_fname = '../unirefsmall/'
dir_arr = ['unirefsmall_16_nn_1_knn_', 'unirefsmall_16_nn_4_knn_']
for i in range(len(dir_arr)):
    dir_arr[i] = nn_classification_fname + dir_arr[i] + '/result.json'

# cls_other = get_cluster_other(dir_learn_on_graph)

cls_arr = []
for i in range(len(dir_arr)):
    cls_tmp = get_cluster_nn_classification(dir_arr[i])
    cls_arr.append(cls_tmp)

# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
# 紫色#b9529f 蓝色#3953a4 红色#ed2024 #231f20 深绿色#098140 浅绿色#7f8133 #0084ff
# solid dotted
# line1_1, = plt.plot(cls_other[0], cls_other[1], marker='X', linestyle='solid', color='#ed2024',
#                     label='Neural LSH')

line2_1, = plt.plot(cls_arr[0][0], cls_arr[0][1], marker='v', linestyle='solid', color='#231f20',
                    label='1 knn')
line2_2, = plt.plot(cls_arr[1][0], cls_arr[1][1], marker='v', linestyle='solid', color='#b9529f',
                    label='4 knn')
# line2_3, = plt.plot(cls_arr[2][0], cls_arr[2][1], marker='v', linestyle='solid', color='#b9529f',
#                     label='8 knn')

# plt.xscale('log')
# plt.xlim(1, 500000)

# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
# plt.title('graph kmeans vs knn')
# plt.legend(loc='upper left', title="SIFT10K, 10-NN, 16 cluster")
plt.legend(loc='lower right', title="uniref1K, 10-NN, 16 cluster")

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()