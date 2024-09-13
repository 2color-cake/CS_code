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

# 5.介数中心性
def cal_betweenessCentrality(L):
    hubs = []     # 存所有图的hubs再取并集
    # 对每个矩阵进行分析
    for i, adjacency_matrix in enumerate(L):
        # 构建网络图
        G = nx.from_numpy_array(adjacency_matrix)

        # 计算节点介数中心性
        betweenness_centrality = nx.betweenness_centrality(G)

        # 找到介数中心性最高的前 个节点
        sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
        hub_nodes = sorted_nodes[:74]
        hub_nodes = [node + 1 for node in hub_nodes]    # 每个+1 因为序号是从1开始

        # 输出“hub”节点
        # print(f"########Graph {i+1}")
        # print(adjacency_matrix)
        # print("Top 53 Hub Nodes:", sorted(hub_nodes))   # 按顺序输出
        # print()
        hubs.append(set(hub_nodes))   ###
    # 计算至少在3/4的图中出现的节点
    hub_nodes_intersection = set()
    for node in hubs[0]:
        count = 1
        for j in range(1, len(L)):
            if node in hubs[j]:
                count += 1
        if count >= len(L) * 1:
            hub_nodes_intersection.add(node)
    hub_nodes_intersection = {x + 1 for x in hub_nodes_intersection}  # 节点序号统一
    return hub_nodes_intersection

# 6.度数中心性 （分别算每一张再求交集）
def cal_degreeCentrality(L):
    degree_centralities = []
    # 遍历矩阵列表L
    for i, matrix in enumerate(L):
        # 构建网络图
        G = nx.from_numpy_array(matrix)
        # 计算节点的度数中心性
        degree_centrality = nx.degree_centrality(G)

        degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        degree_centralities.append(degree_centrality)
        # 输出
        # print(f"########Graph {i + 1}")
        # for node, centrality in degree_centrality:
            # print("节点：", node, "度数中心性：", centrality)  # 按顺序输出

    # 统计节点在图中出现的次数 (只计数最关键的前56个点
    node_counts = {}
    for centrality in degree_centralities:
        for node, _ in centrality[:74]:
            node_counts[node] = node_counts.get(node, 0) + 1

    # 提取在至少3/4的图中都出现过的节点编号
    top_nodes = set(node for node, count in node_counts.items() if count >= 1 * len(L))
    degree_nodes = {x + 1 for x in top_nodes}  # 节点序号统一
    return degree_nodes

# 计算两个列表的交集列表
def cal_intersection(List1, List2):
    set1 = set(List1)
    set2 = set(List2)
    intersection = set1 & set2
    return list(intersection)

def calculate_rich_club(L):
    for matrix in L:
        graph = nx.from_numpy_array(matrix)
        degrees = dict(graph.degree())
        sorted_degrees = sorted(degrees.values(), reverse=True)
        num_nodes = len(graph)

        rich_club = []
        for k in range(1, max(sorted_degrees) + 1):
            nodes = [node for node, degree in degrees.items() if degree >= k]
            subgraph = graph.subgraph(nodes)

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
            plt.title('Rich-club Coefficient')
            plt.grid(True)
            plt.show()

def cal_maxDegree(L):
    for i, adj_matrix in  enumerate(L):
        graph = nx.from_numpy_array(adj_matrix)

        # 计算节点的度数
        degrees = dict(graph.degree())
        # 找到具有最高度数的节点及其度数
        node, degree = max(degrees.items(), key=lambda x: x[1])
        print("Graph", i, "节点", node, "的度数为", degree)

def cal_small_world(L):
    for i, matrix in enumerate(L):
        # 将矩阵转化为图对象
        graph = nx.from_numpy_array(matrix)

        # 计算小世界网络特征
        avg_shortest_path_length = nx.average_shortest_path_length(graph)
        clustering_coefficient = nx.average_clustering(graph)
        # 计算小世界属性
        small_world_coefficient = nx.algorithms.smallworld.sigma(graph, niter=10, nrand=10)

        # 打印小世界属性
        print("Graph", i, "小世界属性：", small_world_coefficient, "平均最短路径长度:", avg_shortest_path_length, "聚集系数:", clustering_coefficient)
        print()

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


# 7.对两种排序内部再进行比较,如分析二者都前多少个里有多少交集
def compare_weight(L56_weight_node_rank, L148_weight_node_rank):
    for i in range(5, len(L56_weight_node_rank)+1, 5):  # 从5到56，步长为5
        L56_subset = set(L56_weight_node_rank[:i])
        L148_subset = set(L148_weight_node_rank[:i])
        intersection = L56_subset.intersection(L148_subset)
        print(f"##########前 {i} 个元素的交集大小：{len(intersection)}\n"
              f"比例: {len(intersection)/i * 100}%\n"
              f"{intersection}")



if __name__ == '__main__':
    # 导入邻接矩阵列表（30个）
    L56 = load_Ldata(L56_path)
    L148 = load_Ldata(L148_path)

    # 1.计算SSIM相似度
    SSIM_53 = cal_SSIM(L56)
    SSIM_148 = cal_SSIM(L148)
    print("########平均SSIM")
    print(f"53 节点共{len(L56)}组:{SSIM_53}")
    print(f"148节点共{len(L148)}组:{SSIM_148}")
    print()

    # 2.计算Pearson相似度
    P_53 = cal_PearsonCorrelation(L56)
    P_148 = cal_PearsonCorrelation(L148)
    print("########平均pearson correlation coefficient")
    print(f"53 节点共{len(L56)}组:{P_53}")
    print(f"148节点共{len(L148)}组:{P_148}")
    print()

    # 3.计算平均cosine similarity
    C_53 = cal_cosineSimilarity(L56)
    C_148 = cal_cosineSimilarity(L148)
    print("########平均cosine similarity")
    print(f"53 节点共{len(L56)}组:{C_53}")
    print(f"148节点共{len(L56)}组:{C_148}")
    print()

    # cal_edge_nums(L56)   #平均边数占比：0.26585884353741496 还比较稳定
    # cal_edge_nums(L148)  #平均边数占比：0.1626491356221086 还比较稳定

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
    L56_weight_node_rank = cal_weight(L56, 56)

    # 7.对两种排序内部再进行比较,如分析二者都前多少个里有多少交集
    compare_weight(L56_weight_node_rank, L148_weight_node_rank)
