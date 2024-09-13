import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap

node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]
node = [2, 11, 19, 21, 22, 26, 27, 38, 42, 44, 46, 51, 56, 57, 58, 59, 60, 65, 67, 68, 69, 73, 76, 81, 85, 86, 88, 93, 94, 95, 96, 99, 100, 101, 103, 104, 110, 114, 116, 118, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 139, 141, 142, 143, 147]
intersection = [2, 11, 19, 21, 22, 26, 27, 44, 56, 58, 65, 67, 68, 69, 73, 81, 85, 93, 94, 95, 96, 99, 100, 101, 103, 104, 116, 118, 127, 128, 130, 132, 139, 141, 142, 147]

L148_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L148"
L56_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L56"


s_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\edge"
f_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\node"
num_node = 148
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

# 1.计算带权度中心性
def calculate_weighted_degree_centrality(weighted_matrix, li):
    # 创建带权图
    G = nx.from_numpy_array(weighted_matrix)

    # 计算每个节点的带权度
    weighted_degrees = {}
    for node in G.nodes():
        weighted_degree = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
        weighted_degrees[node] = weighted_degree

    # 计算带权度中心性
    num_nodes = len(G)
    weighted_degree_centrality = {}
    for node, weighted_degree in weighted_degrees.items():
        weighted_degree_centrality[node] = weighted_degree / (num_nodes - 1) if num_nodes > 1 else 0.0

    # 打印带权度中心性字典
    print(weighted_degree_centrality)

    # 打印指定节点的带权度中心性
    tmp = []
    for i in li:
        tmp.append(weighted_degree_centrality[i-1])
    print(tmp)


# 计算度中心性
def calculate_degree_centrality(binary_matrix):
    # 计算每个节点的度
    degrees = np.sum(binary_matrix, axis=1)
    # 总节点数
    num_nodes = len(binary_matrix)
    # 计算每个节点的度中心性
    degree_centrality = degrees / (num_nodes - 1)

    # print("度中心性:", degree_centrality)
    # 获取排序后的索引
    sorted_indices = np.argsort(degree_centrality)[::-1]
    # 根据排序后的索引获取排序后的度中心性和节点序号
    sorted_degree_centrality = degree_centrality[sorted_indices]
    sorted_node_indices = np.arange(1, len(degree_centrality) + 1)[sorted_indices]
    # 输出排序后的节点序号和度中心性
    print("排序后的节点序号:", sorted_node_indices[:])
    print("排序后的度中心性:", sorted_degree_centrality[:])
    tmp = []
    li = node_index
    for i in li:
        tmp.append(sorted_degree_centrality[list(sorted_node_indices[:]).index(i)])
    print(tmp)

# 2.计算介数中心性
def calculate_betweenness_centrality(weighted_matrix, li):
    # 创建带权图
    G = nx.from_numpy_array(weighted_matrix)
    # 计算介数中心性
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    # 将结果按节点顺序转化为列表
    betweenness_list = [betweenness_centrality[node] for node in sorted(G.nodes())]

    # 打印介数中心性列表
    print(betweenness_list)

    tmp = []
    for i in li:
        tmp.append(betweenness_list[i - 1])
    print(tmp)

# 3.计算closeness中心性
def calculate_closeness_centrality(weighted_matrix, li):
    # 创建带权图
    G = nx.from_numpy_array(weighted_matrix)

    # 计算带权接近中心性
    closeness_centrality = nx.closeness_centrality(G, distance='weight')

    # 将结果按节点顺序转化为列表
    closeness_list = [closeness_centrality[node] for node in sorted(G.nodes())]

    # 打印带权接近中心性列表
    print(closeness_list)
    tmp = []
    for i in li:
        tmp.append(closeness_list[i - 1])
    print(tmp)

# 4.计算eigenvector中心度  特征向量中心度
def calculate_eigenvector_centrality(weighted_matrix, li):
    # 创建带权图
    G = nx.from_numpy_array(weighted_matrix)

    # 计算特征向量中心度
    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

    # 将结果按节点顺序转化为列表
    eigenvector_list = [eigenvector_centrality[node] for node in sorted(G.nodes())]

    # 打印特征向量中心度列表
    print(eigenvector_list)

    tmp = []
    for i in li:
        tmp.append(eigenvector_list[i - 1])
    print(tmp)


def viewer_adj_target_nodes(L, target_nodes):
    from matplotlib.colors import LinearSegmentedColormap
    # 创建一个颜色映射，根据权值大小从白色到蓝色渐变
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue', 'red'])

    # 创建一个与原始矩阵大小相同的矩阵，初始化为白色
    Adj = np.zeros_like(L, dtype=float)

    # 将与目标节点相关的边设置为红色，其他边则根据权值从白色到蓝色渐变
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if L[i, j] != 0:
                if (i + 1 in target_nodes) or (j + 1 in target_nodes):
                    Adj[i, j] = 2 + L[i, j]  # 设置为红色
                else:
                    Adj[i, j] = L[i, j]  # 设置为蓝色（根据权值）

    # 可视化 Adj
    plt.imshow(Adj, cmap=cmap, interpolation='nearest')
    plt.title('Adjacency Matrix with Edges Related to Target Nodes')
    plt.colorbar()
    plt.show()

# sort的是148
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

# sort的是56
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

def cal_weight_edges(sorted_edges_desc , percentage=0.5):
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



if __name__ == '__main__':
    L148 = load_Ldata(L148_path)
    tmp148 = np.sum(L148, axis=0, dtype=np.float64)
    tmp148 = Min_Max_Norm(tmp148)    # 矩阵之和+归一化

    # calculate_weighted_degree_centrality(tmp148)
    # 调用函数，将 tmp 中的数据二值化
    # binary_tmp = binarize_matrix(tmp148, percentage=0.3)
    # calculate_betweenness_centrality(binary_tmp)
    # print(binary_tmp)
    # viewer_adj_single(tmp)
    node_56inner20 = [130, 68, 56, 2, 21, 100, 142, 116, 58, 132, 19, 95, 93, 89, 139, 27, 101, 96, 65, 69]
    sc = load_s_adjmatrix()
    sc = Min_Max_Norm(sc)
    fc = load_f_adjmatrix()
    fc = Min_Max_Norm(fc)
    matrix = fc
    li = node_56inner20
    print("degree")
    calculate_weighted_degree_centrality(matrix, li)
    print("betweenness")
    calculate_betweenness_centrality(matrix, li)
    print("closeness")
    calculate_closeness_centrality(matrix, li)
    print("e")
    calculate_eigenvector_centrality(matrix, li)



    '''
    viewer_adj_target_nodes(tmp148, [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147])
    edge_148 = sort_edges(tmp148)
    cal_weight_edges(edge_148)
    edge_148 = edge_148[:int(0.5*len(edge_148))]

    L56 = load_Ldata(L56_path)
    tmp56 = np.sum(L56, axis=0, dtype=np.float64)
    tmp56 = Min_Max_Norm(tmp56)  # 矩阵之和+归一化
    edge_56 = sort_edges_56(tmp56)
    edge_56 = edge_56[:int(0.5 * len(edge_56))]
    cal_weight_edges(edge_56)
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
    '''
