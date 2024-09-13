import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

num_node = 148

task = "fc"
s_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\edge"
f_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\node"

# 1. 把先验的结构矩阵导入（平均了98个subject的结果）
def load_s_adjmatrix():
    node_num = num_node
    graph_num = 98

    # 文件夹所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(s_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # 存图邻接矩阵的字典
    graphs = {}
    for i in range(graph_num):
        graphs[i] = np.zeros((node_num, node_num), dtype=int)  # 创建邻接矩阵数组

    # 全部存入
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:  # 读取文件
            lines = file.readlines()
        # 解析文件内容
        edge_num = int(lines[0])  # 边的数量
        edges = [list(map(int, line.split())) for line in lines[1:]]  # [[1, 5], [1, 15], [1, 16],.....]的列表
        # 填充邻接矩阵
        for edge in edges:
            node_a, node_b = edge
            graphs[i][node_a - 1, node_b - 1] = 1  # 减1是因为节点序号从1开始
            graphs[i][node_b - 1, node_a - 1] = 1  # 对称

    avg_adjMatrix = np.zeros((node_num, node_num))  # 总的
    # 做平均再可视化
    for i in range(graph_num):
        avg_adjMatrix += graphs[i]
    avg_adjMatrix /= graph_num
    return avg_adjMatrix

# 2. 把先验的直接平均的功能矩阵导入（平均了7类的(已经都平均完了））
def load_f_adjmatrix():
    node_num = num_node
    # 文件夹所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(f_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # print(file_paths)

    matrix = np.zeros((node_num, node_num))
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = [list(map(float, line.strip().split())) for line in lines[1:]]
        matrix += np.array(data)

    # print(len(file_paths))
    adj_function = matrix / len(file_paths)
    # print(adj_function)
    return adj_function


# 3. 返回按权重度值排序的节点排序
def cal_weight(L):
    node_index = np.array(
        [51, 116, 130, 57, 42, 27, 56, 76, 131, 93, 125, 2, 132, 58, 133, 95, 19, 59, 21, 142, 101, 147, 68, 100, 135,
         129, 73, 67, 96, 88, 134, 139, 69, 85, 60, 22, 99, 143, 118, 44, 86, 65, 141, 94, 26, 38, 128, 137, 110, 11,
         127, 81, 103, 46, 114, 104, 25, 15, 102, 7, 120, 29, 123, 20, 145, 89, 52, 115, 14, 112, 40, 126, 50, 108, 84,
         30, 43, 119, 124, 90, 78, 54, 144, 77, 91, 62, 105, 28, 3, 31, 138, 72, 121, 106, 61, 49, 32, 111, 12, 34, 36,
         136, 37, 97, 79, 146, 13, 70, 23, 6, 16, 92, 83, 10, 24, 53, 98, 80, 45, 66, 148, 63, 4, 109, 122, 117, 35,
         113, 1, 82, 71, 8, 87, 18, 140, 48, 55, 75, 64, 5, 47, 17, 107, 39, 74, 9, 33, 41])

    # 按列求和
    column_sums = np.sum(L, axis=0)

    # 按照从高到低排序的列号
    sorted_columns = np.argsort(-column_sums) + 1  # 使用负号实现从高到低排序

    # 输出排序后的列号
    print("Graph_sum_weight_rank : \n", sorted_columns)

    # 计算每个百分比阈值对应的节点数
    percentiles = np.arange(10, 110, 10)
    num_nodes = np.ceil(percentiles / 100 * len(sorted_columns)).astype(int)

    # 统计每个百分比阈值下的节点与 node_index 的交集
    for i in range(len(percentiles)):
        num = num_nodes[i]
        selected_nodes = sorted_columns[:num]
        intersection = np.intersect1d(selected_nodes, node_index)
        non_intersection = set(selected_nodes) - set(intersection)  # 在 selected_nodes 中除了交集之外的部分
        percentage = percentiles[i]

        print(f"######前 {percentage}% 的 sorted_columns 节点中与 node_index 的交集为：\n{intersection} \n"
              f"交集个数为：{len(intersection)}\n"
              f"非交集为：{non_intersection}\n"
              f"非交集个数为：{len(non_intersection)}\n"
              f"占 scale1 节点比例：{len(intersection) / len(node_index) * 100}%\n"
              f"占比：{100 * len(intersection) / (percentage / 100 * len(sorted_columns))}%\n")

# 4. 这个全部view一个矩阵
def viewer_adj(fc):
    # 计算前per%的阈值
    per = 100
    threshold = np.percentile(fc.flatten(), 100-per)

    # 将低于50%阈值的连接设为0
    fc[fc < threshold] = 0
    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'])

    # 可视化sum_result
    plt.imshow(fc, cmap=cmap, interpolation='nearest')
    plt.title(f' {task} Adj')
    plt.colorbar()
    plt.savefig(f'{task}.png')  # 保存图像
    plt.close()  # 关闭图形窗口

def visualize_weighted_degree(matrix):
    node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67,
                  68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127,
                  128, 130, 132, 139, 141, 142, 147]
    intersection_nodes = [2, 11, 19, 21, 22, 26, 27, 44, 56, 58, 65, 67, 68, 69, 73, 81, 85, 93, 94, 95, 96, 99, 100, 101, 103, 104, 116, 118, 127, 128, 130, 132, 139, 141, 142, 147]
    non_intersection_nodes = [38, 42, 46, 51, 57, 59, 60, 76, 86, 88, 110, 114, 125, 129, 131, 133, 134, 135, 137, 143]

    # 计算每个节点的权重度（度的加权和）
    weighted_degrees = np.sum(matrix, axis=0)
    print(weighted_degrees)
    # 获取节点的排序索引
    sorted_indices = np.argsort(weighted_degrees)[::-1]  # 按照权重度从大到小排序
    print(sorted_indices)
    # 根据排序索引获取排序后的权重度
    sorted_weighted_degrees = weighted_degrees[sorted_indices]

    # 绘制直方图
    plt.figure(figsize=(20, 6), dpi=600)  # 增加图形的尺寸
    bars = plt.bar(np.arange(len(sorted_weighted_degrees)), sorted_weighted_degrees, color='#FBDBE9',
                   width=0.8)  # 调整直方图的宽度

    # 突出显示指定节点并添加图例
    legend_handles = []
    legend_labels = ['Non-overlapping Hub', 'Common', 'Specific']

    for node in node_index:
        if node - 1 in sorted_indices:
            index = np.where(sorted_indices == node - 1)[0][0]
            bars[index].set_color('#A4CBF6')
            if not legend_handles:
                legend_handles.append(plt.bar([0], [0], edgecolor="blue", color='#A4CBF6')[0])  # 添加一个虚拟的条形图作为图例
    for node in intersection_nodes:
        if node - 1 in sorted_indices:
            index = np.where(sorted_indices == node - 1)[0][0]
            bars[index].set_color('#B6D095')
            if len(legend_handles) == 1:
                legend_handles.append(plt.bar([0], [0], edgecolor="blue", color='#B6D095')[0])  # 添加一个虚拟的条形图作为图例
    for node in non_intersection_nodes:
        if node - 1 in sorted_indices:
            index = np.where(sorted_indices == node - 1)[0][0]
            bars[index].set_color('#E397A3')
            if len(legend_handles) == 2:
                legend_handles.append(plt.bar([0], [0], edgecolor="blue", color='#E397A3')[0])  # 添加一个虚拟的条形图作为图例

    plt.xlabel('Region')
    plt.ylabel('Functional Connectivity Strength')
    # plt.title('Weighted Degree Distribution')

    # 添加图例
    plt.legend(legend_handles, legend_labels)

    # 增加间隔和旋转标签
    plt.xticks(np.arange(len(sorted_weighted_degrees)), sorted_indices + 1, rotation=90)  # 旋转标签

    plt.tight_layout()  # 自动调整子图参数，以便标签不重叠
    # plt.show()
    plt.savefig("functionconnectivity.jpg")

if __name__ == '__main__':
    sc = load_s_adjmatrix()
    # viewer_adj(sc)
    # print(sc)

    fc = load_f_adjmatrix()
    # cal_weight(fc)
    # viewer_adj(fc)

    # degrees = {node: np.sum(fc[node]) for node in range(len(fc))}  # 示例度值字典
    # # visualize_degree_sorted(fc, degrees)
    visualize_weighted_degree(fc)
    # # print(fc)
