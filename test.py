import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.Graph()  # 创建一个空的无向图
# 节点的个数
num = 6
nodes = list(range(num))  # [0,1,2,3,4,5]
# 将节点添加到网络中
G.add_nodes_from(nodes)  # 从列表中加点
edges = []  # 存放所有的边，构成无向图（去掉最后一个结点，构成一个环）
# edges.append((1, 4))
edges.append((2, 5))

#  将所有边加入网络
G.add_edges_from(edges)
# 每个节点对应坐标坐标
coordinates = [[1, 2], [2, 2], [3, 2], [3, 1], [2, 1], [1, 1]]
# 可导入自己所需的数据
vnode = np.array(coordinates)
npos = dict(zip(nodes, vnode))  # 获取节点与坐标之间的映射关系，用字典表示
# 若显示多个图，可将所有节点放入该列表中
# pos = {}
# pos.update(npos)
nlabels = dict(zip(nodes, nodes))  # 标志字典，构建节点与标识点之间的关系
nx.draw_networkx_nodes(G, npos, node_size=50, node_color="#6CB6FF")  # 绘制节点
nx.draw_networkx_edges(G, npos, edges)  # 绘制边
# nx.draw_networkx_labels(G, npos, nlabels)  # 标签
x_max, y_max = vnode.max(axis=0)  # 获取每一列最大值
x_min, y_min = vnode.min(axis=0)  # 获取每一列最小值
x_num = (x_max - x_min) / 10
y_num = (y_max - y_min) / 10
# print(x_max, y_max, x_min, y_min)
plt.xlim(x_min - x_num, x_max + x_num)
plt.ylim(y_min - y_num, y_max + y_num)
plt.show()
