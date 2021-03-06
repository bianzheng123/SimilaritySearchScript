import json
import matplotlib.pyplot as plt


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


method2category = {
    'opq': 'baseline',
    'pq': 'baseline',
    'knn': 'nn',
    'knn_lsh': 'nn',
    'knn_random_projection': 'nn',
    'knn_kmeans': 'nn',
    'knn_kmeans_multiple': 'nn',
    'e2lsh': 'count'
}

# deep gist glove imagenet sift
dataset_name = 'siftsmall'
n_cluster = 16

method_l = [
    'knn', 'pq', 'knn_random_projection', 'knn_lsh', 'knn_kmeans', 'knn_kmeans_multiple'
    # 'pq', 'knn_random_projection', 'knn_lsh', 'knn_kmeans'
]
n_classifier_l = [
    4
]
dir_arr = []
for n_classifier in n_classifier_l:
    for method in method_l:
        cate = method2category[method]
        fname = '%s_%d_%s_%d_%s' % (dataset_name, n_cluster, cate, n_classifier, method)
        if cate == 'nn' or cate == 'count':
            fname = '%s_' % fname
        print(fname)
        dir_arr.append("../%s/%s/result.json" % (dataset_name, fname))

cls_arr = []
for i in range(len(dir_arr)):
    cls_tmp = get_cluster_nn_classification(dir_arr[i])
    cls_arr.append(cls_tmp)

# 第一个是横坐标的值，第二个是纵坐标的值
# plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
# 紫色#b9529f 蓝色#3953a4 红色#ed2024 #231f20 深绿色#098140 浅绿色#7f8133 #0084ff
# solid dotted

cnt_idx = 0
marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']

linestyle_l = ['dotted', 'solid']
if len(n_classifier_l) == 1:
    linestyle_l = ['solid']
for i, n_classifier in enumerate(n_classifier_l):
    for j, method in enumerate(method_l):
        label = '%d %s' % (n_classifier, method)
        plt.plot(cls_arr[cnt_idx][0], cls_arr[cnt_idx][1], marker=marker_l[j], linestyle=linestyle_l[i],
                 color=color_l[j],
                 label=label)
        cnt_idx += 1

plt.xscale('log')
# plt.xlim(1, 500000)

# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
# plt.title('graph kmeans vs knn')
title_ds_name = '%s %s' % (dataset_name, '10K' if 'small' in dataset_name else '1M')
plt.legend(loc='upper left', title="%s, top-10, %d cluster" % (title_ds_name, n_cluster))

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
