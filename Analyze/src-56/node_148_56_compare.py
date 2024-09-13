import os

import numpy as np
import matplotlib.pyplot as plt

L148_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L148"
L56_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L56"


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
        L.append(Min_Max_Norm(np.matrix(L_data)))  # 归一化后的
        # io.savemat(f"{file_name}.mat", {'array': L_data})

    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            L[i] = np.array(L[i])  # 原本是matrix类型对象，转成ndarray形式
            L[j] = np.array(L[j])
    return L
# L的元素必须是ndarray类型
# 返回按权重排序后的20组节点排序
def cal_weight(L, type):
    if type == 148:
        sum_matrix = L[0]
        for i in range(1, len(L)):
            sum_matrix += L[i]
        # 归一化
        sum_matrix = Min_Max_Norm(sum_matrix)
        # 按列求和,并对求和后的weight做最大最小归一化
        column_sums = np.sum(sum_matrix, axis=0)
        column_sums = (column_sums - np.min(column_sums)) / (np.max(column_sums) - np.min(column_sums))
        # print(column_sums)

        # 创建点和权重的元组列表
        node_weight_list = []
        for i, weight in enumerate(column_sums):
            node_weight_list.append((i + 1, weight))

        # 按权重从高到低排序
        sorted_node_weight = sorted(node_weight_list, key=lambda x: x[1], reverse=True)

        # 输出排序后的节点和权重
        print("Graph_sum_weight_rank : \n", sorted_node_weight)
        return sorted_node_weight

    else:  # 56的时候需要列号的二次转换 +1 以及映射回148全部
        node_index = np.array(
            [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68,
             69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128,
             130, 132, 139, 141, 142, 147])
        node_dictionary = {}  # 把1-56返回原本的对应关系字典里去
        for i, index in enumerate(node_index):
            node_dictionary[index] = i
        node_dictionary = {value: key for key, value in node_dictionary.items()}
        print(node_dictionary)
        sum_matrix = L[0]
        for i in range(1, len(L)):
            sum_matrix += L[i]
        sum_matrix = Min_Max_Norm(sum_matrix)
        # 按列求和
        column_sums = np.sum(sum_matrix, axis=0)
        column_sums = (column_sums - np.min(column_sums)) / (np.max(column_sums) - np.min(column_sums))
        # 创建点和权重的元组列表
        node_weight_list = []
        for i, weight in enumerate(column_sums):
            node_weight_list.append((node_dictionary[i], weight))

        # 按权重从高到低排序
        sorted_node_weight = sorted(node_weight_list, key=lambda x: x[1], reverse=True)

        # 输出排序后的节点和权重
        print("Graph_sum_weight_rank : \n", sorted_node_weight)
        return sorted_node_weight


# 无实际权重，仅形式上表现
def draw_node_compare():
    ## node排序占比
    # 已有按照权重排好序的节点序号列表 node_148 和 node_56
    node_148 = [51, 116, 130, 57, 42, 27, 56, 76, 131, 93, 125, 2, 132, 58, 133, 95, 19, 59, 21, 142, 101, 147, 68, 100,
                135, 129, 73, 67, 96, 88, 134, 139, 69, 85, 60, 22, 99, 143, 118, 44, 86, 65, 141, 94, 26, 38, 128, 137,
                110, 11, 127, 81, 103, 46, 114, 104, 25, 15, 102, 7, 120, 29, 123, 20, 145, 89, 52, 115, 14, 112, 40,
                126, 50, 108, 84, 30, 43, 119, 124, 90, 78, 54, 144, 77, 91, 62, 105, 28, 3, 31, 138, 72, 121, 106, 61,
                49, 32, 111, 12, 34, 36, 136, 37, 97, 79, 146, 13, 70, 23, 6, 16, 92, 83, 10, 24, 53, 98, 80, 45, 66,
                148, 63, 4, 109, 122, 117, 35, 113, 1, 82, 71, 8, 87, 18, 140, 48, 55, 75, 64, 5, 47, 17, 107, 39, 74,
                9, 33, 41]
    node_56 = [56, 130, 2, 116, 21, 68, 100, 58, 142, 19, 132, 27, 95, 139, 65, 93, 89, 101, 96, 99, 85, 7, 104, 69, 67,
               81, 128, 26, 44, 22, 147, 127, 11, 52, 25, 30, 103, 118, 54, 73, 15, 102, 141, 20, 29, 53, 80, 126, 94,
               28, 4, 90, 45, 6, 119, 16]

    # 创建柱状图的高度列表
    bar_heights = []

    # 遍历 node_148 中的节点序号
    for node in node_148:
        # 如果节点在 node_56 中出现过，则将该节点在 node_56 中的索引作为柱状图的高度
        if node in node_56:
            bar_heights.append(56 - node_56.index(node) + 1)
        # 如果节点不在 node_56 中，则将高度设为0
        else:
            bar_heights.append(0)
    print(bar_heights)
    # 不转换的话只会自动按顺序1-148
    node_148_str = [str(i) for i in node_148]
    # 绘制柱状图
    plt.figure(figsize=(16, 6))
    plt.bar(node_148_str, bar_heights, edgecolor='blue', linewidth=1, color='skyblue')
    # 增加间隔和旋转标签
    plt.xticks(np.arange(len(node_148)), rotation=90)  # 旋转标签

    # 添加标签和标题
    plt.xlabel('Node Index (Sorted by node_148)')
    plt.ylabel('Node Weight Order (Sorted by node_56)')
    plt.title('Comparison of Node Weights between node_148 and node_56')

    # 显示图形
    plt.tight_layout()
    plt.show()

# 有实际权重，横轴坐标为node_148的顺序，按照权重148的权重趋势画成折线，56的画成条形，高度为权重
def draw_node_compare_weight(node_56, node_148):
    # 将节点序号和权重分离
    node_56_indices, node_56_weights = zip(*node_56)
    node_148_indices, node_148_weights = zip(*node_148)

    # 创建柱状图的高度列表
    bar_heights = []

    # 遍历 node_148 中的节点序号
    for node in node_148_indices:
        # 如果节点在 node_56 中出现过，则将该节点在 node_56 中的权重作为柱状图的高度
        if node in node_56_indices:
            bar_heights.append(node_56_weights[node_56_indices.index(node)])
        # 如果节点不在 node_56 中，则将高度设为0
        else:
            bar_heights.append(0)

    # 不转换的话只会自动按顺序1-148
    node_148_str = [str(i) for i in node_148_indices]

    # 绘制柱状图
    plt.figure(figsize=(16, 6))
    bars = plt.bar(node_148_str, bar_heights, edgecolor='blue', linewidth=1, color='lightskyblue')

    # 绘制折线图
    line, = plt.plot(node_148_str, node_148_weights, marker='.', color='r', linewidth=1)

    # 增加间隔和旋转标签
    plt.xticks(np.arange(len(node_148_indices)), rotation=90)

    # 添加标签和标题
    plt.xlabel('Node Index (Sorted by node_148)')
    plt.ylabel('Node Weight')
    plt.title('Comparison of Node Weights between node_148 and node_56')

    # 添加图例
    plt.legend([bars[0], line], ['node_56', 'node_148'])

    # 显示图形
    plt.tight_layout()
    plt.show()


L148 = load_Ldata(L148_path)
node_148 = cal_weight(L148, type=148)
L56 = load_Ldata(L56_path)
node_56 = cal_weight(L56, type=56)
draw_node_compare_weight(node_56, node_148)
