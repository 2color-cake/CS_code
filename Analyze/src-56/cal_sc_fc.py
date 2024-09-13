import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

num_node = 148

task = "sc"
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

# 3. 全部view一个矩阵
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

# 4.与FC做对比
def visualize_weighted_degree(matrix):
    node_56 = [56, 130, 2, 116, 21, 68, 100, 58, 142, 19, 132, 27, 95, 139, 65, 93, 89, 101, 96, 99, 85, 7, 104, 69, 67,
               81, 128, 26, 44, 22, 147, 127, 11, 52, 25, 30, 103, 118, 54, 73, 15, 102, 141, 20, 29, 53, 80, 126, 94,
               28, 4, 90, 45, 6, 119, 16]

    # 计算每个节点的权重度（度的加权和）
    weighted_degrees = np.sum(matrix, axis=0)
    print(weighted_degrees)
    # 获取节点的排序索引
    sorted_indices = np.argsort(weighted_degrees)[::-1]  # 按照权重度从大到小排序
    node_dictionary = {}  # 把1-56返回原本的对应关系字典里去
    node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67,
                  68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127,
                  128, 130, 132, 139, 141, 142, 147]
    # for i, index in enumerate(node_index):
    #     node_dictionary[index] = i
    # # print(node_dictionary)
    # node_dictionary = {value: key for key, value in node_dictionary.items()}
    # print(node_dictionary)
    # print(sorted_indices)
    # sorted_indices = [node_dictionary[i] for i in sorted_indices]
    # print(sorted_indices)
    # 创建柱状图的高度列表
    bar_heights = []

    # 遍历 node_148 中的节点序号
    for node in sorted_indices:
        # 如果节点在 node_56 中出现过，则将该节点在 node_56 中的索引作为柱状图的高度
        if node+1 in node_56:
            bar_heights.append(56 - node_56.index(node+1) + 1)
        # 如果节点不在 node_56 中，则将高度设为0
        else:
            bar_heights.append(0)
    print(bar_heights)
    # 不转换的话只会自动按顺序1-148
    node_148_str = [str(i+1) for i in sorted_indices]
    # 绘制柱状图
    plt.figure(figsize=(16, 6))
    plt.bar(node_148_str, bar_heights, edgecolor='blue', linewidth=1, color='skyblue')
    # 增加间隔和旋转标签
    plt.xticks(np.arange(len(sorted_indices)), rotation=90)  # 旋转标签

    # 添加标签和标题
    plt.xlabel('Node Index (Sorted by fc_56)')
    plt.ylabel('Node Weight Order (Sorted by node_56)')
    plt.title('Comparison of Node Weights between fc and node_56')

    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sc = load_s_adjmatrix()
    # viewer_adj(sc)
    # print(sc)

    fc = load_f_adjmatrix()
    # viewer_adj(fc)

    visualize_weighted_degree(fc)

