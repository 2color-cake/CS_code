import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap

L148_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L148"
L56_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L56"
node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]

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


# 这个是选特定点view # 特定点可视化
def viewer_adj_target_nodes(L, target_nodes):
    # print("共可视化相关节点"len(target_nodes))
    # 创建一个全零矩阵
    Adj = np.zeros((148, 148))
    Adj2 = np.zeros((56, 56))

    dic = {}
    for i, n in enumerate(node_index):
        dic[n] = i

    # 将与目标节点相关的所有边保留
    for node_i in range(148):
        for node_j in range(148):
            if node_i + 1 in target_nodes and node_j + 1 in target_nodes:
                if L[node_i, node_j] != 0:
                    Adj[node_i, node_j] = L[node_i, node_j]
                    Adj2[dic[node_i + 1], dic[node_j + 1]] = L[node_i, node_j]

    # 可视化 Adj
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'])
    plt.imshow(Adj2, cmap=cmap, interpolation='nearest')
    plt.title('Adjacency Matrix with Edges Related to Target Nodes')
    plt.colorbar()
    plt.show()
    return Adj2

if __name__ == '__main__':
    L148 = load_Ldata(L148_path)
    Adj_56 = []
    for l in L148:
        Adj_56.append(viewer_adj_target_nodes(l, target_nodes=node_index))
    print("SSIM: ", cal_SSIM(Adj_56))
    print("PC: ", cal_PearsonCorrelation(Adj_56))
    print("CC: ", cal_cosineSimilarity(Adj_56))

    tmp148 = np.sum(L148, axis=0, dtype=np.float64)
    tmp148 = Min_Max_Norm(tmp148)  # 矩阵之和+归一化
