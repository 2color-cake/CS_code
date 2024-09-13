import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
from scipy.stats import pearsonr


L56_path = r""
L148_path = r"EL-L148"
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

# 计算两个列表的交集列表
def cal_intersection(List1, List2):
    set1 = set(List1)
    set2 = set(List2)
    intersection = set1 & set2
    return list(intersection)


# L的元素必须是ndarray类型
# 返回按权重排序后的20组节点排序
def cal_weight(L, type):
    node_index = np.array([2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68,
              69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128,
              130, 132, 139, 141, 142, 147])
    if type == 148:
        sum_matrix = L[0]
        for i in range(1, len(L)):
            sum_matrix += L[i]
        # 按列求和
        column_sums = np.sum(sum_matrix, axis=0)
        # 按照从高到低排序的列号
        sorted_columns = np.argsort(-column_sums)  # 使用负号实现从高到低排序
        sorted_columns += 1
        # 输出排序后的列号
        print("Graph_sum_weight_rank : \n", sorted_columns)

        sorted_columns = np.array(sorted_columns)

        # 计算每个百分比阈值对应的节点数
        percentiles = np.arange(10, 110, 10)
        num_nodes = np.ceil(percentiles / 100 * len(sorted_columns)).astype(int)

        # 统计每个百分比阈值下的节点与node_index的交集
        for i in range(len(percentiles)):
            num = num_nodes[i]
            selected_nodes = sorted_columns[:num]
            intersection = np.intersect1d(selected_nodes, node_index)
            percentage = percentiles[i]
            # print(sorted_columns.shape[0])
            print(f"######前 {percentage}% 的 sorted_columns 节点中与 node_index 的交集为：\n{intersection} \n"
                  f"个数为：{len(intersection)}\n"
                  f"占scale1节点比:{len(intersection) / len(node_index) * 100}%\n"
                  f"占比：{100 * len(intersection) / (percentage / 100 * sorted_columns.shape[0])}%\n")
            # f"细分T0个数：{len(set(intersection) & set(node_vote_T0))}\n"
            # f"细分T012个数：{len(set(intersection) & set(node_vote_T012))}\n")

        return sorted_columns

    else:  # 56的时候需要列号的二次转换 +1 以及映射回148全部
        node_dictionary = {}  # 把1-53返回原本的对应关系字典里去
        for i, index in enumerate(node_index):
            node_dictionary[index] = i + 1
        node_dictionary = {value: key for key, value in node_dictionary.items()}
        print(node_dictionary)
        sum_matrix = L[0]
        for i in range(1, len(L)):
            sum_matrix += L[i]
        # 按列求和
        column_sums = np.sum(sum_matrix, axis=0)
        # 按照从高到低排序的列号
        sorted_columns = np.argsort(-column_sums)  # 使用负号实现从高到低排序
        sorted_columns += 1
        for i, node in enumerate(sorted_columns):
            sorted_columns[i] = node_dictionary[node]

        # 输出排序后的列号
        print("53 Graph_sum_weight_rank : \n", sorted_columns)

        return sorted_columns




if __name__ == '__main__':
    # 导入邻接矩阵列表（30个）
    # L56 = load_Ldata(L56_path)
    L148 = load_Ldata(L148_path)



    # 6.直接统计权重情况
    # 统计列表里每个矩阵的情况
    L148_weight_node_rank = cal_weight(L148, 148)
    # L56_weight_node_rank = cal_weight(L56, 56)

