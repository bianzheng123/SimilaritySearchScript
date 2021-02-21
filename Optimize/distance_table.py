import numpy as np
import time


def count_centroid(base, centroid_l):
    # count the distance for each item and centroid to get the distance_table
    distance_table = None
    for i, vecs in enumerate(base, 0):
        if i == 0:
            distance_table = [np.linalg.norm(base[0] - centroid) for centroid in centroid_l]
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
    data = np.random.normal(size=100000).reshape(-1, 2)
    centroid = np.random.normal(size=100).reshape(-1, 2)
    start_time = time.time()
    labels = count_centroid(data, centroid)
    end_time = time.time()
    np.savetxt('label.txt', labels, fmt='%d')
    print('time consumed %.3f' % (end_time - start_time))
