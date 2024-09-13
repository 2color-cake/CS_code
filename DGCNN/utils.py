import networkx as nx  # 导入NetworkX库，用于处理图数据
import numpy as np  # 导入NumPy库，用于数值计算

import math  # 导入Math库，用于数学计算
from torch.utils.data import Dataset    # 用于生成数据集

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

"""Adapted from https://github.com/weihua916/powerful-gnns/blob/master/util.py"""

# 定义一个S2VGraph类，用于表示图数据的结构
# 暂时扔掉了拓扑结构（没用上）
class S2VGraph(object):
    def __init__(self, g, node_features):
        '''
            g: 一个NetworkX图对象，表示图的拓扑结构
            neighbors: 邻居列表（不包括自环）
            node_features: 一个Torch张量，节点特征的one-hot表示，用作神经网络的输入
            edge_mat: 一个Torch长整型张量，包含边列表，将用于创建Torch稀疏张量
            max_neighbor存最大度
        '''
        self.g = g
        self.node_features = node_features


# 定义加载数据的函数load_data
def load_data(dataset, file_list, c_i):
    '''
        dataset: 数据集的全部路径         ["D:\DataSet\DS\\node\\EMOTION\\Pears_148_obj_",
                                   A       "D:\DataSet\DS\\node\\WM\\Pears_148_obj_",
                                   B       "D:\DataSet\DS\edge\\structure_148_edge_unweight_obj_"]
        file_list: 包含数据文件列表的文件路径  C （是list.txt，里面是所有图的名称号，与edge和node的文件名称最后的位置对应）
        c_i: 当前数据集在全部路径的索引
    '''

    print('loading data')
    g_list = []  # 存储图数据的列表

    file_node = dataset[c_i]
    with open(file_list, 'r') as f:
        num_list = f.readline().strip().split()

    for i in range(len(num_list)):
        name_file_node = (file_node + '%d.txt' % int(num_list[i]))
        g = nx.Graph()  # 创建一个NetworkX图对象
        node_features = []

        with open(name_file_node, 'r') as f:
            n_node = int(f.readline().strip())
            for j in range(n_node):
                g.add_node(j)  # 添加节点到图（节点读的就是从0-147,编号也就这么编了）

                row = f.readline().strip().split()
                attr = np.array(row, dtype=np.float32)
                node_features.append(attr)  # 将节点特征添加到列表中

        #print(node_features)
        #print(np.array(node_features))
        g_list.append(S2VGraph(g, np.array(node_features)))  # 将图数据构造为S2VGraph对象并添加到列表中
        #exit(0)

    print("# data: %d" % len(g_list))
    return g_list

# 定义数据分割函数separate_data （总共分了10折，fold_idx只是用来定义前多少折用于训练的）
def separate_data(graph_list, fold_idx, seed=0):
    assert 1 <= fold_idx and fold_idx < 10, "fold_idx must be from 1 to 10."

    one_sample = math.floor(len(graph_list) / 10)
    index = [i for i in range(0, len(graph_list))]
    np.random.seed(123)
    np.random.shuffle(index)
    train_idx = index[0:one_sample * fold_idx]
    test_idx = index[one_sample * fold_idx: len(index)]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_idx, test_idx, train_graph_list, test_graph_list

# 邻接矩阵可视化并存储
def save_adjmatrix_L(adj_matrix, save_path):
    adj_matrix_cpu = adj_matrix.detach().cpu().numpy()

    # 计算阈值，保留值最大的前 1/10 的点
    #threshold = np.percentile(adj_matrix_cpu, 99)

    # 将低于阈值的值设置为 NaN 进行遮罩
    #adj_matrix_cpu[adj_matrix_cpu < threshold] = np.nan
    adj_matrix_cpu[adj_matrix_cpu == 0] = np.nan

    cmap = LinearSegmentedColormap.from_list('custom', [(0, 'white'), (1, 'blue')])
    plt.figure(figsize=(15, 10), dpi=500)
    ax = sns.heatmap(adj_matrix_cpu, cmap=cmap, linecolor='white', linewidths=0.1, square=True, mask=np.isnan(adj_matrix_cpu))

    num_nodes = adj_matrix_cpu.shape[0]
    step = 3
    ax.set_xticks(range(0, num_nodes, step))
    ax.set_yticks(range(0, num_nodes, step))
    ax.set_xticklabels(range(1, num_nodes + 1, step))
    ax.set_yticklabels(range(1, num_nodes + 1, step))

    plt.title('Adjacency Matrix')
    plt.xlabel('Nodes')
    plt.ylabel('Nodes')
    plt.savefig(save_path)
    plt.close()

# 继承自Dataset类并实现len和getitem方法的 导入fMRI数据集
class fMRIDataSet(Dataset):
    def __init__(self, data, label):   # 数据集，标签
        self.data = data
        self.label = label

    def __getitem__(self, index):      # 同时返回：对应index的data和其label
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)          # 返回数据集的长度
