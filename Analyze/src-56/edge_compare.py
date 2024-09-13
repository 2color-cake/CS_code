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
        L.append( Min_Max_Norm(np.matrix(L_data)) )    # 归一化后的
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
    weight = sorted_values[-num_edges_to_keep]    # 倒过来
    print(f"保留前{percentage}数量边的邻接权重值: ", weight)

    num_rows, num_cols = matrix.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] >= weight:
                matrix[i][j] = matrix[i][j]      ############ 1
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
    for idx in sorted_indices_desc[: int(1*len(sorted_indices_desc))]:   # 取前10%的边
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
    node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]
    for i, index in enumerate(node_index):
        node_dictionary[index] = i + 1
    node_dictionary = {value: key for key, value in node_dictionary.items()}
    # print(node_dictionary)
    # 根据排序后的索引构造排序后的边和权重列表
    sorted_edges_desc = []
    for idx in sorted_indices_desc[: int(1*len(sorted_indices_desc))]:   # 取前10%的边
        row, col = divmod(idx, adj_matrix.shape[1])

        # 将节点标签加上1
        node1 = node_dictionary[row+1]
        node2 = node_dictionary[col+1]
        # 确保添加 (a, b) 和 (b, a) 中较小的索引作为边
        edge = (min(node1, node2), max(node1, node2))

        # 检查边是否已经添加，如果没有则添加，并记录在集合中
        if edge not in added_edges and weights[idx] != 0:
            sorted_edges_desc.append((edge[0], edge[1], weights[idx]))
            added_edges.add(edge)
    # print(sorted_edges_desc)
    return sorted_edges_desc

def cal_weight_edges(sorted_edges_desc , percentage):
    # 需要计算前percentage的边中，含node_index列表节点的不同边数
    # 1. 两点均为node_index中节点 2.只有一点是 3.两点都不是
    # sorted_edges_desc = [(99, 99, 1.0), (50, 73, 0.9182400893926537), (80, 80, 0.9173286433112159)]三元组列表
    node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]
    # 计算需要考虑的边数
    num_edges = int(len(sorted_edges_desc) * percentage)
    # print(sorted_edges_desc[:num_edges])
    # 初始化计数器
    count_both = 0  # 两点均为node_index中节点
    count_single = 0  # 只有一点是node_index中节点
    count_none = 0  # 两点都不是node_index中节点


    # 遍历前percentage比例的边
    for i in range(num_edges):
        edge = sorted_edges_desc[i]
        # 判断边的两个节点是否在node_index列表中
        if edge[0] in node_index and edge[1] in node_index:
            count_both += 1
        elif edge[0] in node_index or edge[1] in node_index:
            count_single += 1
        else:
            count_none += 1
    print(f"前{percentage}总边数: ", num_edges)
    print("两点均为Scale-1节点的边数:", count_both)
    print("只有一点是Scale-1节点的边数:", count_single)
    print("两点都不是Scale-1中节点的边数:", count_none)

def draw_overlap_matrix(edge_148, edge_56):
    # edge_148的边显示为蓝色，edge_56的边显示为粉色，重叠的边设置为紫色
    # 创建一个全零矩阵作为初始矩阵
    matrix = np.zeros((148, 148))

    # 将 edge_148 的边设置为蓝色（值为1）
    for edge in edge_148:
        matrix[edge[0] - 1, edge[1] - 1] = 1
        matrix[edge[1] - 1, edge[0] - 1] = 1

    # 将 edge_56 的边设置为粉色（值为2）
    for edge in edge_56:
        matrix[edge[0] - 1, edge[1] - 1] = 2
        matrix[edge[1] - 1, edge[0] - 1] = 2

    # 将 edge_148 和 edge_56 重叠的边设置为紫色（值为3）
    for edge in edge_148:
        if (edge[0], edge[1]) in [(e[0], e[1]) for e in edge_56]:
            # if edge[2] >= 0.5:
               matrix[edge[0] - 1, edge[1] - 1] = 3
               matrix[edge[1] - 1, edge[0] - 1] = 3

    # 绘制矩阵
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Edge Type')
    plt.title('Overlap Matrix of Edge_148 and Edge_56')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.show()


if __name__ == '__main__':
    L148 = load_Ldata(L148_path)
    tmp148 = np.sum(L148, axis=0, dtype=np.float64)
    tmp148 = Min_Max_Norm(tmp148)    # 矩阵之和+归一化
    edge_148 = sort_edges(tmp148)
    # print(edge_148[:20])


    percentage_148 = 1
    edge_148 = edge_148[:int(percentage_148*len(edge_148))]
    # edge_148 = edge_148[:5690]
    cal_weight_edges(edge_148, percentage=1)

    L56 = load_Ldata(L56_path)
    tmp56 = np.sum(L56, axis=0, dtype=np.float64)
    tmp56 = Min_Max_Norm(tmp56)  # 矩阵之和+归一化
    edge_56 = sort_edges_56(tmp56)
    percentage_56 = 1
    edge_56 = edge_56[:int(percentage_56 * len(edge_56))]
    # edge_56 = edge_56[:1000]

    draw_overlap_matrix(edge_148, edge_56)

    print(len(edge_148))
    print(len(edge_56))

    cnt = 0
    l1 = []
    for a in edge_148:
        for b in edge_56:
            if a[0] == b[0] and a[1] == b[1]:
                cnt += 1
                l1.append(a)
                l1.append(b)
    print(cnt)
    print(l1)

