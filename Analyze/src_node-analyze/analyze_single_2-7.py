import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap

L56_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L56"
L148_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L148"   # 120个图稳定性多判断几个
task = L148_path.split("\\")[5]
L_init_xavier_normal_path = r"D:\Projects\model\DS_generate-rest_task\L_init_xavier_normal.txt"

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

# 绘制度数直方图
def draw_degreeHistogram_figure(adjMatrix):
    # 计算度数
    degrees = np.sum(adjMatrix > 0, axis=1)  # 基于非零权重值计算度数

    # 设置权重值范围和区间数
    weight_min = np.min(adjMatrix)
    weight_max = np.max(adjMatrix)
    num_bins = 10  # 指定直方图的区间数

    # 统计度数频率
    hist, bins = np.histogram(adjMatrix, bins=num_bins, range=(weight_min, weight_max))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 绘制直方图
    plt.bar(bin_centers, hist, width=(weight_max - weight_min) / num_bins)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Degree Histogram')
    plt.show()

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

# 简单统计一下边数占比等情况
def cal_edge_nums(L):
    count_all = 0
    for l in L:
        count = np.count_nonzero(l)
        print(f"边数：{count}")
        count_all += count
    print(f"平均边数：{count_all/len(L)}")
    print(f"平均边数占比：{count_all/(len(L) * (L[0].size))}")

# 相似程度
# 1.计算矩阵列表L里所有的SSIM
def cal_SSIM(L):
    SSIM_all = 0
    for i in range(len(L)):
        for j in range(i+1, len(L)):
            # print(ssim(L[i], L[j], data_range=1.0))
            SSIM_all += ssim(L[i], L[j], data_range=1.0)
            print(ssim(L[i], L[j], data_range=1.0))
    SSIM_avg = SSIM_all/math.comb(len(L), 2)   # C(n,2)种
    return SSIM_avg

# 2.计算矩阵列表L里所有的皮尔逊相关性
def cal_PearsonCorrelation(L):
    P_all = 0
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            L[i] = np.array(L[i])    # 原本是matrix类型对象，转成ndarray形式才能正常被flatten,从而在后续计算皮尔逊相关性
            L[j] = np.array(L[j])
            # print(len(L[i].flatten()))
            # print(len(L[j].flatten()))
            P = pearsonr(L[i].flatten(), L[j].flatten())[0]
            print(P)
            # print(P)
            P_all += P
    P_avg = P_all / math.comb(len(L), 2)  # C(n,2)种
    return P_avg

# 3.算余弦相似度
def cal_cosineSimilarity(L):
    from scipy.spatial.distance import cosine
    C_all = 0
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            L[i] = np.array(L[i])  # 原本是matrix类型对象，转成ndarray形式才能正常被flatten
            L[j] = np.array(L[j])
            # print(len(L[i].flatten()))
            # print(len(L[j].flatten()))
            C = 1 - cosine(L[i].flatten(), L[j].flatten())
            # print(C)
            C_all += C
            print(C)
    C_avg = C_all / math.comb(len(L), 2)  # C(n,2)种
    return C_avg

# 4.计算边的分布情况（按照出现次数）（还是比较随机）
def Edge_Distribution(L):
    for adjacency_matrix in L:
        # 步骤1：提取所有有权值的边（即所有边）
        num_edges = adjacency_matrix.size   # 53*53 = 2809 / 148*148
        # print(num_edges)

        sorted_edges = np.sort(adjacency_matrix, axis=None)[::-1]   # 按权值高低排的列表
        # print(sorted_edges)

        threshold = sorted_edges[int(0.1 * num_edges)]   # 取全部边的前%个

        # threshold = 0   # 取全部边
        selected_edges = (adjacency_matrix > threshold).astype(int) # 步骤2：将所选的边转化为0-1形式
        # print(selected_edges)


        # 步骤3：统计总体的边情况
        if 'total_edges' not in locals():
            total_edges = selected_edges
        else:
            total_edges += selected_edges
    np.set_printoptions(threshold=np.inf)
    print(total_edges)

    # 将total_edges转换为一维数组
    flat_total_edges = total_edges.flatten()

    # 创建直方图
    plt.hist(flat_total_edges, bins=range(14), align='left', rwidth=0.8, density=True)
    plt.xlabel('Edge Count')
    plt.ylabel('Frequency')
    plt.title('Edge Distribution')
    plt.show()

    # total_edges 现在包含了总体的边情况，其中大于1的值表示边出现在多个矩阵中。

# 5.介数中心性((可选保留多少边）)
def cal_betweenessCentrality(matrix):
    per = 50
    # 计算前per%的边的权重阈值
    threshold = np.percentile(matrix.flatten(), 100 - per)

    # 保留权重大于阈值的边
    new_matrix = np.where(matrix > threshold, matrix, 0)
    # 构建网络图
    G = nx.from_numpy_array(new_matrix)

    # 计算节点介数中心性
    betweenness_centrality = nx.betweenness_centrality(G)

    # 按度数中心性排序
    sorted_centrality = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

    # 输出
    print(f"########Graph {task}")
    for node, centrality in sorted_centrality:
        print("节点：", node + 1, "介数中心性：", centrality, end=" ")

# 6.度数中心性(可选保留多少边）
def cal_degreeCentrality(matrix):
    per = 50
    # 计算前per%的边的权重阈值
    threshold = np.percentile(matrix.flatten(), 100 - per)

    # 保留权重大于阈值的边
    new_matrix = np.where(matrix > threshold, matrix, 0)

    # 构建网络图
    G = nx.from_numpy_array(new_matrix)
    # 计算节点的度数中心性
    degree_centrality = nx.degree_centrality(G)

    # 按度数中心性排序
    sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    # 输出
    print(f"########Graph {task}")
    for node, centrality in sorted_centrality:
        print("节点：", node+1, "度数中心性：", centrality, end=" ")


# 7. 单个rich_club
def calculate_rich_club(matrix):
    def remove_self_loops(graph):
        # 移除自环
        graph_without_self_loops = graph.copy()
        graph_without_self_loops.remove_edges_from(nx.selfloop_edges(graph))
        return graph_without_self_loops
    per = 100
    # 计算前per%的边的权重阈值
    threshold = np.percentile(matrix.flatten(), 100 - per)

    # 保留权重大于阈值的边
    new_matrix = np.where(matrix > threshold, matrix, 0)

    # 构建带权图
    weighted_graph = nx.from_numpy_array(new_matrix)
    weighted_graph = remove_self_loops(weighted_graph)  # 移除自环

    degrees = dict(weighted_graph.degree())
    sorted_degrees = sorted(degrees.values(), reverse=True)
    rich_club = []
    for k in range(1, max(sorted_degrees) + 1):
        nodes = [node for node, degree in degrees.items() if degree >= k]
        subgraph = weighted_graph.subgraph(nodes)

        total_edges = subgraph.number_of_edges()
        max_possible_edges = (len(nodes) * (len(nodes) - 1)) / 2

        if max_possible_edges > 0:
            normalized_edges = total_edges / max_possible_edges
            rich_club.append((k, normalized_edges))

    # 提取k值和rich-club系数
    k_values = [item[0] for item in rich_club]
    rich_club_coef = [item[1] for item in rich_club]

    # 绘制rich-club系数图形
    plt.plot(k_values, rich_club_coef, marker='o')
    plt.xlabel('k')
    plt.ylabel('Rich-club Coefficient')
    plt.title(f'Rich-club Coefficient (Top {per}%)')
    plt.grid(True)
    plt.show()

    graph = weighted_graph
    target_degree = 100
    # 提取大于等于目标度的节点
    rich_club_nodes = [node for node in graph.nodes() if graph.degree(node) >= target_degree]

    # 构建富裕俱乐部子图
    rich_club_subgraph = graph.subgraph(rich_club_nodes)

    # 可视化富裕俱乐部，使用random布局
    pos = nx.random_layout(rich_club_subgraph)

    # 为节点添加标签，并将标签的值加1
    labels = {node: f'{node + 1}' for node in rich_club_subgraph.nodes()}
    nodes = []
    for node in rich_club_subgraph.nodes():
        nodes.append(node+1)
    print()
    print(nodes)

    nx.draw(rich_club_subgraph, pos, with_labels=True, labels=labels, node_size=50)
    plt.title(f'Rich-club Subgraph (Degree >= {target_degree})')
    plt.show()


# 8. 单个small-world
def cal_small_world(matrix):
    per = 50
    # 计算前per%的边的权重阈值
    threshold = np.percentile(matrix.flatten(), 100 - per)

    # 保留权重大于阈值的边
    new_matrix = np.where(matrix > threshold, matrix, 0)

    # 将矩阵转化为图对象
    graph = nx.from_numpy_array(new_matrix)
    print(graph.degree)
    print(graph.nodes)
    print(graph.edges)

    # 计算小世界网络特征
    avg_shortest_path_length = nx.average_shortest_path_length(graph)
    clustering_coefficient = nx.average_clustering(graph)
    # 计算小世界属性
    # small_world_coefficient = nx.algorithms.smallworld.sigma(graph, niter=10, nrand=10)

    # 打印小世界属性
    print("Graph ", task, "平均最短路径长度:", avg_shortest_path_length,
          "聚集系数:", clustering_coefficient)
    print()

# L的元素必须是ndarray类型
# 返回按权重排序后的20组节点排序
def cal_weight(L, type):
    node_index = np.array([2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58,
                           65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116,
                           118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147])
    if type == 148:
        sum_matrix = L[0]
        for i in range(1, len(L)):
            sum_matrix += L[i]
        # 按列求和
        column_sums = np.sum(sum_matrix, axis=0)
        print(list(column_sums))
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
    elif type == 1481:
        select_56 = [2, 11, 19, 21, 22, 26, 27, 38, 42, 44, 46, 51, 56, 57, 58, 59, 60, 65, 67, 68, 69, 73, 76, 81, 85,
                     86, 88, 93, 94, 95, 96, 99, 100, 101, 103, 104, 110, 114, 116, 118, 125, 127, 128, 129, 130, 131,
                     132, 133, 134, 135, 137, 139, 141, 142, 143, 147]
        sum_matrix = L[0]
        for i in range(1, len(L)):
            sum_matrix += L[i]
        # 按列求和
        column_sums = np.sum(sum_matrix, axis=0)
        column_sums = Min_Max_Norm(column_sums)
        print(column_sums)
        weight_sum = []
        for i in range(len(column_sums)):
            if (i+1) in node_index:
                weight_sum.append(column_sums[i])
        print(weight_sum)

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
        column_sums = Min_Max_Norm(column_sums)
        print(sorted(column_sums, reverse=True))
        # 按照从高到低排序的列号
        sorted_columns = np.argsort(-column_sums)  # 使用负号实现从高到低排序
        sorted_columns += 1
        for i, node in enumerate(sorted_columns):
            sorted_columns[i] = node_dictionary[node]

        # 输出排序后的列号
        print("56 Graph_sum_weight_rank : \n", sorted_columns)

        return sorted_columns


# 7.对两种排序内部再进行比较,如分析二者都前多少个里有多少交集
def compare_weight(L56_weight_node_rank, L148_weight_node_rank):
    for i in range(5, len(L56_weight_node_rank)+1, 5):  # 从5到56，步长为5
        L56_subset = set(L56_weight_node_rank[:i])
        L148_subset = set(L148_weight_node_rank[:i])
        intersection = L56_subset.intersection(L148_subset)
        print(f"##########前 {i} 个元素的交集大小：{len(intersection)}\n"
              f"比例: {len(intersection)/i * 100}%\n"
              f"{intersection}")

# 0. 邻接矩阵分析前:和的稳定性验证（单个不稳定，群体和稳定也能说明一定问题）
def sum_stablity(L, type):
    # 初始化存储结果的列表
    sum_results = []

    for i in range(0, 30, 5):  # 步长为5，从5到30
        # 选择前i个邻接矩阵
        selected_matrices = L[i:i+10]
        # 计算选择的邻接矩阵的和
        sum_result = np.sum(selected_matrices, axis=0, dtype=np.float64)
        sum_result = Min_Max_Norm(sum_result)   # 归一化

        # 存储结果
        sum_results.append(sum_result)

    # 1.计算SSIM相似度
    print(sum_results)
    SSIM = cal_SSIM(sum_results)
    print("########平均SSIM")
    print(f"{type}节点共{len(sum_results)}组:{SSIM}")
    print()

    # 2.计算Pearson相似度
    P = cal_PearsonCorrelation(sum_results)
    print("########平均pearson correlation coefficient")
    print(f"{type}节点共{len(sum_results)}组:{P}")
    print()

    # 3.计算平均cosine similarity
    # C_56 = cal_cosineSimilarity(L56)
    C = cal_cosineSimilarity(sum_results)
    print("########平均cosine similarity")
    print(f"{type}节点共{len(sum_results)}组:{C}")
    print()

# 这个全部view
def viewer_adj(L, type):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    sum_result = np.sum(L, axis=0, dtype=np.float64)
    sum_result = Min_Max_Norm(sum_result)   # 归一化
    print(sum_result)
    if type == 56:
        # 节点映射关系
        node_mapping = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65,
                        67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119,
                        126, 127, 128, 130, 132, 139, 141, 142, 147]

        # 创建一个 148*148 的零矩阵
        Adj148 = np.zeros((148, 148))

        # 遍历 L56 矩阵，将其值填充到 Adj148 对应的位置上
        for i in range(56):
            for j in range(56):
                node_i = node_mapping[i] - 1  # 节点映射关系中是从 1 开始的，而数组是从 0 开始的
                node_j = node_mapping[j] - 1
                Adj148[node_i, node_j] = sum_result[i, j]
        sum_result = Adj148
        print(sum_result)
    # 计算前per%的阈值
    per = 100
    threshold = np.percentile(sum_result.flatten(), 100-per)

    # 将低于50%阈值的连接设为0
    sum_result[sum_result < threshold] = 0


    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'])

    # 可视化sum_result
    plt.imshow(sum_result, cmap=cmap, interpolation='nearest')
    plt.title(f' {task} Adj')
    plt.colorbar()
    plt.savefig(f'{task}.png')  # 保存图像
    plt.close()  # 关闭图形窗口

# 这个是选特定点view # 特定点可视化
def viewer_adj_target_nodes(L, target_nodes):
    # print("共可视化相关节点"len(target_nodes))
    # 创建一个全零矩阵
    Adj = np.zeros((148, 148))

    # 将与目标节点相关的所有边保留
    for node_i in range(148):
        for node_j in range(148):
            if node_i + 1 in target_nodes and node_j + 1 in target_nodes:
                if L[node_i, node_j] != 0:
                    Adj[node_i, node_j] = L[node_i, node_j]
            elif (node_i + 1 in target_nodes and node_j + 1 not in target_nodes) or \
                 (node_i + 1 not in target_nodes and node_j + 1 in target_nodes):
                if L[node_i, node_j] != 0:
                    Adj[node_i, node_j] = L[node_i, node_j]

    # 可视化 Adj
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'])
    plt.imshow(Adj, cmap=cmap, interpolation='nearest')
    plt.title('Adjacency Matrix with Edges Related to Target Nodes')
    plt.colorbar()
    plt.show()


# 显示单张矩阵
def viewer_adj_single(L, name):
    from matplotlib.colors import LinearSegmentedColormap
    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'])

    # 可视化sum_result
    plt.imshow(L, cmap=cmap, interpolation='nearest')
    plt.title(f'{name} Adj')
    plt.colorbar()
    plt.savefig(f'{name}.png')  # 保存图像
    plt.close()  # 关闭图形窗口

# L的元素必须是ndarray类型
# 要做算RBO的，因此添加了很多需要输出单个值的
def cal_weight_RBO(L, type):
    node_index = np.array([2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68,
              69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128,
              130, 132, 139, 141, 142, 147])
    if type == 148:
        for i in range(0, len(L)):
            # 按列求和
            column_sums = np.sum(L[i], axis=0)
            # 按照从高到低排序的列号
            sorted_columns = np.argsort(-column_sums)  # 使用负号实现从高到低排序
            sorted_columns += 1
            # 输出排序后的列号
            print(f"np.array({list(sorted_columns)}),")

    else:  # 56的时候需要列号的二次转换 +1 以及映射回148全部
        node_dictionary = {}  # 把1-53返回原本的对应关系字典里去
        for i, index in enumerate(node_index):
            node_dictionary[index] = i + 1
        node_dictionary = {value: key for key, value in node_dictionary.items()}
        print(node_dictionary)
        for i in range(0, len(L)):
            # 按列求和
            column_sums = np.sum(L[i], axis=0)
            # 按照从高到低排序的列号
            sorted_columns = np.argsort(-column_sums)  # 使用负号实现从高到低排序
            sorted_columns += 1
            for i, node in enumerate(sorted_columns):
                sorted_columns[i] = node_dictionary[node]

            # 输出排序后的列号
            print("56 Graph_sum_weight_rank : \n", sorted_columns)

        return sorted_columns

s_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\edge"

# 1. 把先验的结构矩阵导入（平均了98个subject的结果）
def load_s_adjmatrix():
    node_num = 148
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
def load_f_adjmatrix(num_node = 148):
    f_folder_path = r"D:\DataSet\DS_task_rest\task\DS_148\node"
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

if __name__ == '__main__':
    # 导入邻接矩阵列表（30个）
    L56 = load_Ldata(L56_path)
    L148 = load_Ldata(L148_path)

    '''
    L_init_xavier_normal = np.loadtxt(L_init_xavier_normal_path)
    viewer_adj_single(L_init_xavier_normal, "L_init_xavier_normal")
    '''

    # # 0. 验证邻接矩阵之和的稳定性
    # sum_stablity(L56, 56)
    # sum_stablity(L148, 148)
    # viewer_adj(L56, 56)
    # viewer_adj(L148, 148)

    # tmp = np.sum(L148, axis=0, dtype=np.float64)
    # tmp = Min_Max_Norm(tmp)
    # fc = load_f_adjmatrix()
    # sc = load_s_adjmatrix()
    # fc = Min_Max_Norm(fc)
    # viewer_adj(tmp, 148)
    # calculate_rich_club(fc)
    # cal_weight(L148, 1481)    # 1481新写的为了算一下平均权重
    # cal_weight(L56, 56)
    #     # 4. 度数中心性衡量
    #     cal_degreeCentrality(tmp)
    #     draw_degreeHistogram_figure(tmp)
    #     # 5.介数中心性衡量
    #     cal_betweenessCentrality(tmp)
    #     # 6.rich-club
    #     calculate_rich_club(tmp)
    #     # 7. small world
    #     # cal_small_world(tmp)
    #     viewer_adj_target_nodes(tmp, target_nodes=[2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68,
    #               69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128,
    #               130, 132, 139, 141, 142, 147])

        # # 1.计算SSIM相似度
        # SSIM_53 = cal_SSIM(L56)
        # # SSIM_148 = cal_SSIM(L148)
        # print("########平均SSIM")
        # print(f"53 节点共{len(L56)}组:{SSIM_53}")
        # # print(f"148节点共{len(L148)}组:{SSIM_148}")
        # print()
        #
        # # 2.计算Pearson相似度
        # P_53 = cal_PearsonCorrelation(L56)
        # # P_148 = cal_PearsonCorrelation(L148)
        # print("########平均pearson correlation coefficient")
        # print(f"53 节点共{len(L56)}组:{P_53}")
        # # print(f"148节点共{len(L148)}组:{P_148}")
        # print()
        #
        # # 3.计算平均cosine similarity
        # C_56 = cal_cosineSimilarity(L56)
        # # C_148 = cal_cosineSimilarity(L148)
        # print("########平均cosine similarity")
        # print(f"53 节点共{len(L56)}组:{C_56}")
        # # print(f"148节点共{len(L148)}组:{C_148}")
        # print()
    '''
        # cal_edge_nums(L56)   #平均边数占比：0.26585884353741496 还比较稳定
        cal_edge_nums(L148)  #平均边数占比：0.1626491356221086 还比较稳定
    
        # Edge_Distribution(L56)
        # Edge_Distribution(L148)
    
        # 4. 度数中心性衡量
        degree_nodes = cal_degreeCentrality(L148)
        print("########度数中心性：\n交集中的节点数量：", len(degree_nodes))
        print("交集中的节点编号：")
        print(sorted(degree_nodes))
        degree_ist = cal_intersection(node_index, degree_nodes)
        print("是重要节点的有", len(degree_ist), "个: ", sorted(degree_ist))
        print()
    
        # 5.介数中心性衡量
        betweeness_nodes = cal_betweenessCentrality(L148)
        print("########介数中心性：\n交集中的节点数量：", len(betweeness_nodes))
        print("交集中的节点编号：")
        print(sorted(betweeness_nodes))
        betweeness_ist = cal_intersection(node_index, betweeness_nodes)
        print("是重要节点的有", len(betweeness_ist), "个: ", sorted(betweeness_ist))
        print()
    
    
        # 6.直接统计权重情况（我是蠢猪0.0很绕）
        # 统计列表里每个矩阵的情况
        L148_weight_node_rank = cal_weight(L148, 148)
        # L56_weight_node_rank = cal_weight(L56, 56)
          L148_weight_node_rank = cal_weight(L148, 148)
        # 7.对两种排序内部再进行比较,如分析二者都前多少个里有多少交集
        # compare_weight(L56_weight_node_rank, L148_weight_node_rank)
    '''
    cal_weight(L148, 148)
    # cal_weight(L56, 56)
