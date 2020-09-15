import numpy as np

# 用于解析vecs后缀的文件
# np.set_printoptions(threshold=np.inf)  # 打印numpy数组时显示全部的内容


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy()


base_data = fvecs_read("siftsmall/siftsmall_base.fvecs")
print(base_data)
