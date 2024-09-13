import numpy as np
import matplotlib.pyplot as plt
import os

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


# 二值化
def binarize_matrix(matrix, percentage):
    # 将矩阵展平并按权重值进行降序排序
    sorted_values = np.sort(matrix.flatten())
    # print(sorted_values)  是升序的value
    # 确定需要保留的边的数量（总边数的percentage）
    num_edges_to_keep = int(len(sorted_values) * percentage)
    # 计算前percentage数量边的权重值
    weight = sorted_values[-num_edges_to_keep]  # 倒过来
    print(f"保留前{percentage}数量边的邻接权重值: ", weight)

    num_rows, num_cols = matrix.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] >= weight:
                matrix[i][j] = matrix[i][j]  ############ 1
            else:
                matrix[i][j] = 0
    return matrix


# 显示单张矩阵
def viewer_adj_single(L):
    from matplotlib.colors import LinearSegmentedColormap
    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'])

    # 可视化sum_result
    plt.imshow(L, cmap=cmap, interpolation='nearest')
    plt.title(f'1')
    plt.colorbar()
    plt.show()
    # plt.savefig(f'{name}.png')  # 保存图像
    # plt.close()  # 关闭图形窗口


# sort的是148的边
def sort_edges(adj_matrix):
    # 获取所有边的权重
    weights = adj_matrix.flatten()

    # 获取排序后的索引，按照权重从大到小排序
    sorted_indices_desc = np.argsort(weights)[::-1]

    # 使用集合来跟踪已经添加的边，确保每条边只添加一次
    added_edges = set()

    # 根据排序后的索引构造排序后的边和权重列表
    sorted_edges_desc = []
    for idx in sorted_indices_desc[: int(1 * len(sorted_indices_desc))]:  # 取前10%的边
        row, col = divmod(idx, adj_matrix.shape[1])

        # 将节点标签加上1
        node1 = row + 1
        node2 = col + 1

        # 确保添加 (a, b) 和 (b, a) 中较小的索引作为边
        edge = (min(node1, node2), max(node1, node2))

        # 检查边是否已经添加，如果没有则添加，并记录在集合中
        if edge not in added_edges and weights[idx] != 0:
            sorted_edges_desc.append((edge[0], edge[1], weights[idx]))
            added_edges.add(edge)
    # print(sorted_edges_desc)
    return sorted_edges_desc


# sort的是56的边
def sort_edges_56(adj_matrix):
    # 获取所有边的权重
    weights = adj_matrix.flatten()

    # 获取排序后的索引，按照权重从大到小排序
    sorted_indices_desc = np.argsort(weights)[::-1]

    # 使用集合来跟踪已经添加的边，确保每条边只添加一次
    added_edges = set()
    node_dictionary = {}  # 把1-56返回原本的对应关系字典里去
    node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67,
                  68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127,
                  128, 130, 132, 139, 141, 142, 147]
    for i, index in enumerate(node_index):
        node_dictionary[index] = i + 1
    node_dictionary = {value: key for key, value in node_dictionary.items()}
    # print(node_dictionary)
    # 根据排序后的索引构造排序后的边和权重列表
    sorted_edges_desc = []
    for idx in sorted_indices_desc[: int(1 * len(sorted_indices_desc))]:  # 取前10%的边
        row, col = divmod(idx, adj_matrix.shape[1])

        # 将节点标签加上1
        node1 = node_dictionary[row + 1]
        node2 = node_dictionary[col + 1]
        # 确保添加 (a, b) 和 (b, a) 中较小的索引作为边
        edge = (min(node1, node2), max(node1, node2))

        # 检查边是否已经添加，如果没有则添加，并记录在集合中
        if edge not in added_edges and weights[idx] != 0:
            sorted_edges_desc.append((edge[0], edge[1], weights[idx]))
            added_edges.add(edge)
    # print(sorted_edges_desc)
    return sorted_edges_desc

def cal_weight_edges(sorted_edges_desc, percentage):
    # 需要计算前percentage的边中，含node_index列表节点的不同边数
    # 1. 两点均为node_index中节点 2.只有一点是 3.两点都不是
    # sorted_edges_desc = [(99, 99, 1.0), (50, 73, 0.9182400893926537), (80, 80, 0.9173286433112159)]三元组列表
    node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67,
                  68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127,
                  128, 130, 132, 139, 141, 142, 147]
    # 计算需要考虑的边数
    num_edges = int(len(sorted_edges_desc) * percentage)
    # print(sorted_edges_desc[:num_edges])
    # 初始化计数器
    count_both = 0  # 两点均为node_index中节点
    strength_both = 0
    count_single = 0  # 只有一点是node_index中节点
    strength_single = 0
    count_none = 0  # 两点都不是node_index中节点
    strength_none = 0

    # 遍历前percentage比例的边
    for i in range(num_edges):
        edge = sorted_edges_desc[i]
        # 判断边的两个节点是否在node_index列表中
        if edge[0] in node_index and edge[1] in node_index:
            count_both += 1
            strength_both += edge[2]
        elif edge[0] in node_index or edge[1] in node_index:
            count_single += 1
            strength_single += edge[2]
        else:
            count_none += 1
            strength_none += edge[2]

    print(f"#####前{percentage}总边数: ", num_edges)
    # print("总强度： ", strength_both + strength_single + strength_none)
    print("平均连接强度： ", (strength_both + strength_single + strength_none)/(count_both + count_single + count_none))
    print("#####两点均为Scale-1节点的边数:", count_both)
    # print("总强度： ", strength_both)
    print("平均连接强度： ", strength_both / count_both)
    print("#####只有一点是Scale-1节点的边数:", count_single)
    # print("总强度： ", strength_single)
    print("平均连接强度： ", strength_single / count_single)
    print("#####两点都不是Scale-1中节点的边数:", count_none)
    # print("总强度： ", strength_none)
    print("平均连接强度： ", strength_none / count_none)

num_node = 148

task = "sc"
s_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\edge"
f_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\node"

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

def draw_edge_strength():
    # 提供的数据
    ratios = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    total_edge_strength = [0.878, 0.714, 0.507, 0.432, 0.384, 0.348, 0.317, 0.291, 0.267, 0.245, 0.224, 0.203]
    both_scale1_edge_strength = [0.895, 0.751, 0.509, 0.431, 0.386, 0.354, 0.324, 0.302, 0.281, 0.263, 0.249, 0.238]
    one_scale1_edge_strength = [0.868, 0.701, 0.503, 0.432, 0.383, 0.346, 0.318, 0.292, 0.270, 0.250, 0.232, 0.216]
    no_scale1_edge_strength = [0.864, 0.709, 0.511, 0.435, 0.385, 0.347, 0.313, 0.282, 0.255, 0.229, 0.203, 0.176]

    # 设置柱子的宽度
    bar_width = 0.25

    # 计算柱子的位置
    bar_positions = np.arange(len(ratios))

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    plt.bar(bar_positions - bar_width, both_scale1_edge_strength, edgecolor="blue", width=bar_width, label='Both Scale-1 Nodes',
            color='royalblue')
    plt.bar(bar_positions, one_scale1_edge_strength, edgecolor="blue", width=bar_width, label='One Scale-1 Node', color='yellowgreen')
    plt.bar(bar_positions + bar_width, no_scale1_edge_strength, edgecolor="blue", width=bar_width, label='No Scale-1 Nodes',
            color='lemonchiffon')

    # 添加标签和标题
    plt.xlabel('Ratio')
    plt.ylabel('Average Edge Strength')
    plt.title('Average Edge Strength for Different Ratios')
    plt.xticks(bar_positions, ratios)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    L148 = load_Ldata(L148_path)
    tmp148 = np.sum(L148, axis=0, dtype=np.float64)
    tmp148 = Min_Max_Norm(tmp148)  # 矩阵之和+归一化
    edge_148 = sort_edges(tmp148)
    percentage_148 = 1
    # edge_148 = edge_148[:int(percentage_148 * len(edge_148))]
    # edge_148 = edge_148[:5690]
    cal_weight_edges(edge_148, percentage=percentage_148)

    print()
    fc = load_f_adjmatrix()
    edge_fc = sort_edges(fc)
    cal_weight_edges(edge_fc, percentage=percentage_148)

    draw_edge_strength()
