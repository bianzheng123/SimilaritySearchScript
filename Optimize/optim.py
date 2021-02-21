import numpy as np
import time
from multiprocessing import Pool


def count_centroid(base, centroid_l, idx, pool_size):
    # count the distance for each item and centroid to get the distance_table
    print("start", idx)
    distance_table = None
    for i in range(idx, len(base), pool_size):
        vecs = base[i]
        if i == idx:
            distance_table = [np.linalg.norm(base[i] - centroid) for centroid in centroid_l]
            distance_table = np.array([distance_table])
            continue
        tmp_dis = [np.linalg.norm(vecs - centroid) for centroid in centroid_l]
        tmp_dis = np.array([tmp_dis])
        distance_table = np.append(distance_table, tmp_dis, axis=0)

    # print(distance_table.shape)
    # get the nearest centroid and use it as the label
    labels = np.argmin(distance_table, axis=1)
    return labels


if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.normal(size=100000).reshape(-1, 2)  # 未优化前用了158.024秒 100000
    centroid_l = np.random.normal(size=100).reshape(-1, 2)
    start_time = time.time()

    n_process = 4
    p = Pool(3)
    res_l = []
    for i in range(n_process):
        res = p.apply_async(count_centroid, args=(data, centroid_l, i, n_process))
        res_l.append(res)

    p.close()
    p.join()
    res_labels = np.zeros(data.shape[0]).astype(np.int)
    for i, res in enumerate(res_l, 0):
        tmp_labels = res.get()
        for j in range(len(tmp_labels)):
            res_labels[i + j * n_process] = tmp_labels[i]
        print(res.get().shape)
    np.savetxt('optim_label.txt', res_labels, fmt='%d')
    end_time = time.time()
    print('time consumed %.3f' % (end_time - start_time))
