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
    'knn_random_projection': 'nn',
    'e2lsh': 'count'
}

# deep gist glove imagenet sift
dataset_name = 'sift'
n_cluster = 256

method = 'knn_random_projection'
n_classifier = 4
specific_name = 'model'
specific_val_l = ['one_block_512_dim', 'one_block_2048_dim', 'res_net', 'two_block_512_dim', 'two_block_1024_dim']

dir_arr = []
for val in specific_val_l:
    cate = method2category[method]
    fname = '%s_%d_%s_%d_%s' % (dataset_name, n_cluster, cate, n_classifier, method)
    if cate == 'baseline':
        raise Exception('not support the category')
    fname = '{}_{}_{}'.format(fname, specific_name, val)
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

marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']
for i, val in enumerate(specific_val_l):
    label = val
    plt.plot(cls_arr[i][0], cls_arr[i][1], marker=marker_l[i], linestyle='solid',
             color=color_l[i],
             label=label)

plt.xscale('log')
# plt.xlim(1, 500000)

# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
# plt.title('graph kmeans vs knn')
title_ds_name = '%s %s' % (dataset_name, '10K' if 'small' in dataset_name else '1M')
plt.legend(loc='upper left', title="%s, top-10, %d cluster, %d classifier, parameter: %s" % (title_ds_name, n_cluster, n_classifier, specific_name))

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
