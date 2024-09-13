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


def row_correlation(matrix1, matrix2):
    # 获取矩阵的行数
    num_rows = matrix1.shape[0]

    # 创建一个空的相关性矩阵
    correlation_matrix = np.zeros((num_rows, num_rows))

    # 计算每一对行向量之间的相关性
    for i in range(num_rows):
        for j in range(num_rows):
            # 提取第i行和第j行
            row1 = matrix1[i, :]
            row2 = matrix2[i, :]

            # 计算相关系数（这里使用皮尔逊相关系数）
            correlation_matrix[i, j] = np.corrcoef(row1, row2)[0, 1]

    return correlation_matrix
# 显示单张矩阵
def viewer_adj_single(L):
    from matplotlib.colors import LinearSegmentedColormap
    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'])

    # 可视化sum_result
    plt.imshow(L, cmap="viridis", interpolation='nearest')
    plt.title(f'Pearson Correlation Matrix between fc&trained adj')
    plt.colorbar()
    plt.show()
    # plt.savefig(f'{name}.png')  # 保存图像
    # plt.close()  # 关闭图形窗口



if __name__ == '__main__':
    L148 = load_Ldata(L148_path)
    tmp148 = np.sum(L148, axis=0, dtype=np.float64)
    tmp148 = Min_Max_Norm(tmp148)    # 矩阵之和+归一化
    fc = load_f_adjmatrix()
    fc = Min_Max_Norm(fc)
    sc = load_s_adjmatrix()
    sc = Min_Max_Norm(sc)

    L56 = load_Ldata(L56_path)
    tmp56 = np.sum(L56, axis=0, dtype=np.float64)
    tmp56 = Min_Max_Norm(tmp56)  # 矩阵之和+归一化

    # 计算相关性矩阵
    correlation_matrix = row_correlation(tmp148, fc)


    # 打印相关性矩阵
    print(correlation_matrix)
    viewer_adj_single(correlation_matrix)
