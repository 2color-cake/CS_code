import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
from scipy.stats import pearsonr


L56_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L56"
L148_path = r"D:\Projects\model\DS_generate-rest_task\7-L148"
# 56版重要节点序号
node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68,
              69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128,
              130, 132, 139, 141, 142, 147]

# 最小最大归一化矩阵到[0,1]
def Min_Max_Norm(matrix):
    # Min-Max normalization
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

# 把所有生成的的邻接矩阵数据全都load到一个列表L里
def load_Ldata(L_path):
    # 文件夹所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(L_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    # print(file_paths)
    L = []
    for file_name in file_paths:
        L_data = np.loadtxt(file_name)
        L.append( Min_Max_Norm(np.matrix(L_data)) )    # 归一化后的
        # io.savemat(f"{file_name}.mat", {'array': L_data})

    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            L[i] = np.array(L[i])  # 原本是matrix类型对象，转成ndarray形式
            L[j] = np.array(L[j])
    return L


def sort_edges(adj_matrix):
    adj_matrix = Min_Max_Norm(adj_matrix)   # 还是先做归一化把
    # 获取所有边的权重
    weights = adj_matrix.flatten()

    # 获取排序后的索引，按照权重从大到小排序
    sorted_indices_desc = np.argsort(weights)[::-1]

    # 使用集合来跟踪已经添加的边，确保每条边只添加一次
    added_edges = set()

    # 根据排序后的索引构造排序后的边和权重列表
    sorted_edges_desc = []
    for idx in sorted_indices_desc[: int(0.01*len(sorted_indices_desc))]:   # 只取前10%的边
        row, col = divmod(idx, adj_matrix.shape[1])

        # 将节点标签加上1
        node1 = row + 1
        node2 = col + 1

        # 确保添加 (a, b) 和 (b, a) 中较小的索引作为边
        edge = (min(node1, node2), max(node1, node2))

        # 检查边是否已经添加，如果没有则添加，并记录在集合中
        if edge not in added_edges:
            sorted_edges_desc.append((edge[0], edge[1], weights[idx]))
            added_edges.add(edge)

    return sorted_edges_desc

def import_data_6tasks():
    Set56 = []
    Set148 = []
    for i in range(2, 8):
        # 导入邻接矩阵列表（30个）
        L56 =  load_Ldata(f"D:\Projects\model\DS_generate-rest_task\data\\{i}-L56")
        L148 = load_Ldata(f"D:\Projects\model\DS_generate-rest_task\data\\{i}-L148")

        # 加和得到最大的矩阵（边）
        Adj56 = np.sum(L56, axis=0, dtype=np.float64)
        Adj148 = np.sum(L148, axis=0, dtype=np.float64)

        Set56.append(Adj56)
        Set148.append(Adj148)

    return Set56, Set148



if __name__ == "__main__":
    Set56, Set148 = import_data_6tasks()

    Edges_148 = []
    for j, i in enumerate(Set148):
        sorted_edges_148 = sort_edges(i)
        Edges_148.append(sorted_edges_148)
        # 打印排序后的边
        print(f"########{j+2}分类：{len(sorted_edges_148)} 条边", )
        print(sorted_edges_148)


    # 取第一个元组列表的前两个元素作为初始交集
    intersection_set = set((x[0], x[1]) for x in Edges_148[0])

    # 逐个求取交集
    for edges_list in Edges_148[1:]:
        intersection_set = intersection_set.intersection((x[0], x[1]) for x in edges_list)

    # 将交集结果转换为列表
    intersection_list = list(intersection_set)

    print(f"#######交集边数为：{len(intersection_list)}:", sorted(intersection_list))

    '''
    【保留权重版】
    # 取第一个元组列表的前两个元素作为初始交集
    intersection_set = {(x[0], x[1]): x[2] for x in Edges_148[0]}

    # 逐个求取交集
    for edges_list in Edges_148[1:]:
        new_intersection_set = {}
        for x in edges_list:
            key = (x[0], x[1])
            if key in intersection_set:
                new_intersection_set[key] = intersection_set[key] + x[2]

        intersection_set = new_intersection_set

    # 将交集结果转换为三元组的列表
    intersection_list = [(key[0], key[1], value) for key, value in intersection_set.items()]

    print(intersection_list)
    '''