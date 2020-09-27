import numpy as np
import struct


# 用于解析vecs后缀的文件
# np.set_printoptions(threshold=np.inf)  # 打印numpy数组时显示全部的内容


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d


def fvecs_read(fname):
    data, d = ivecs_read(fname)
    return data.view('float32'), d


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy(), d


# 将文件的一部分加入缓存, 防止文件过大导致加载很慢
def fvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:], d


def bvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:], d


def ivecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.reshape(-1, d + 1)[:, 1:], d


# 将文件以vecs的形式保存
def fvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))  # *dimension就是int, dimension就是list
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def ivecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def read_all(global_path, base_file_name, gnd_file_name, learn_file_name, query_file_name):
    # read from the file
    base_data, vector_dim = fvecs_read(global_path + base_file_name)
    groundtruth_data, gnd_dim = ivecs_read(global_path + gnd_file_name)
    learn_data = fvecs_read(global_path + learn_file_name)[0]
    query_data = fvecs_read(global_path + query_file_name)[0]
    return base_data, groundtruth_data, learn_data, query_data, vector_dim, gnd_dim

# global_path = '/home/bz/SIFT/siftsmall/'
# base_data = fvecs_read_mmap(global_path + 'siftsmall_base.fvecs')
# fvecs_write(global_path + 'test.fvecs', base_data)
# test_data = fvecs_read(global_path + 'test.fvecs')
# print(test_data)
