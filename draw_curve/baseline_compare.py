import json
import matplotlib.pyplot as plt
import os


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


def curve_diff_dataset(method2category, ds_name, method_l, n_cluster, n_classifier):
    dir_arr = []
    for method in method_l:
        cate = method2category[method]
        fname = '%s_%d_%s_%d_%s' % (ds_name, n_cluster, cate, n_classifier, method)
        if cate == 'nn' or cate == 'count':
            fname = '%s_' % fname
        print(fname)
        dir_arr.append("../%s/%s/result.json" % (ds_name, fname))

    cls_arr = []
    for i in range(len(dir_arr)):
        cls_tmp = get_cluster_nn_classification(dir_arr[i])
        cls_arr.append(cls_tmp)

    # marker
    # o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
    # 紫色#b9529f 蓝色#3953a4 红色#ed2024 #231f20 深绿色#098140 浅绿色#7f8133 #0084ff
    # solid dotted

    marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
    color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']

    for i, method in enumerate(method_l):
        label = '%d %s' % (n_classifier, method)
        # 第一个是横坐标的值，第二个是纵坐标的值
        plt.plot(cls_arr[i][0], cls_arr[i][1], marker=marker_l[i], linestyle='solid',
                 color=color_l[i],
                 label=label)

    plt.xscale('log')

    # plt.title('graph kmeans vs knn')
    title_ds_name = '%s %s' % (ds_name, '1M')
    plt.legend(loc='upper left', title="%s, top-10, %d cluster" % (title_ds_name, n_cluster))

    plt.xlabel("the number of candidates")
    plt.ylabel("Recall")
    plt.grid(True, linestyle='-.')
    plt.savefig('./curve/%s_%d.png' % (ds_name, n_classifier))
    plt.close()


def curve_diff_n_classifier(method2category, ds_name, method, n_cluster, n_classifier_l):
    dir_arr = []
    for n_classifier in n_classifier_l:
        cate = method2category[method]
        fname = '%s_%d_%s_%d_%s' % (ds_name, n_cluster, cate, n_classifier, method)
        if cate == 'nn' or cate == 'count':
            fname = '%s_' % fname
        print(fname)
        dir_arr.append("../%s/%s/result.json" % (ds_name, fname))

    cls_arr = []
    for i in range(len(dir_arr)):
        cls_tmp = get_cluster_nn_classification(dir_arr[i])
        cls_arr.append(cls_tmp)

    # marker
    # o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
    # 紫色#b9529f 蓝色#3953a4 红色#ed2024 #231f20 深绿色#098140 浅绿色#7f8133 #0084ff
    # solid dotted

    marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
    color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']

    for i, n_classifier in enumerate(n_classifier_l):
        label = '%d %s' % (n_classifier, method)
        # 第一个是横坐标的值，第二个是纵坐标的值
        plt.plot(cls_arr[i][0], cls_arr[i][1], marker=marker_l[i], linestyle='solid',
                 color=color_l[i],
                 label=label)

    plt.xscale('log')

    # plt.title('graph kmeans vs knn')
    title_ds_name = '%s %s' % (ds_name, '1M')
    plt.legend(loc='upper left', title="%s, top-10, %d cluster" % (title_ds_name, n_cluster))

    plt.xlabel("the number of candidates")
    plt.ylabel("Recall")
    plt.grid(True, linestyle='-.')
    plt.savefig('./curve/%s_diff_classifier.png' % ds_name)
    plt.close()


if __name__ == '__main__':
    method2category = {
        'opq': 'baseline',
        'pq': 'baseline',
        'knn': 'nn',
        'e2lsh': 'count'
    }

    # deep gist glove imagenet sift
    ds_name_l = ['deep', 'gist', 'glove', 'imagenet', 'sift']
    n_cluster = 256

    method_l = [
        'pq', 'e2lsh', 'knn'
    ]
    n_classifier_l = [
        1, 2, 4, 8
    ]
    os.system('mkdir curve')
    for ds_name in ds_name_l:
        for n_classifier in n_classifier_l:
            curve_diff_dataset(method2category, ds_name, method_l, n_cluster, n_classifier)
    for ds_name in ds_name_l:
        curve_diff_n_classifier(method2category, ds_name, 'knn', n_cluster, n_classifier_l)
