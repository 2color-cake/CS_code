from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import math

import numpy as np
from scipy.stats import pearsonr

L148_path = r"D:\Projects\model\DS_generate-rest_task\data"
L56_path = r"D:\Projects\model\DS_generate-rest_task\data"


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
        L.append(np.matrix(L_data))  # 归一化后的
        # io.savemat(f"{file_name}.mat", {'array': L_data})

    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            L[i] = np.array(L[i])  # 原本是matrix类型对象，转成ndarray形式
            L[j] = np.array(L[j])
    return L

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
            P = np.corrcoef(L[i].flatten(), L[j].flatten())[0, 1]
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

def row_correlation(matrix1, matrix2):
    from scipy.spatial.distance import cosine
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
            correlation_matrix[i, j] = 1 - cosine(row1, row2)

    return correlation_matrix
# 显示单张矩阵
def viewer_adj_single(L):
    from matplotlib.colors import LinearSegmentedColormap
    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'])

    # 可视化sum_result
    plt.imshow(L, cmap=cmap, interpolation='nearest')
    plt.title(f'Pearson Correlation Matrix between fc&trained adj')
    plt.colorbar()
    plt.show()
    # plt.savefig(f'{name}.png')  # 保存图像
    # plt.close()  # 关闭图形窗口

if __name__ == '__main__':
    L148 = []   # 六个任务下生成的加和矩阵
    for i in range(2, 8):
        L148_path_ = L56_path + f"\{i}-L148"
        l = load_Ldata(L148_path_)
        viewer_adj_single(l[0])
        tmp148 = np.sum(l, axis=0, dtype=np.float64)
        tmp148 = Min_Max_Norm(tmp148)    # 矩阵之和+归一化
        L148.append(tmp148)
    print(L148)
    print("SSIM: ", cal_SSIM(L148))
    print("PC: ", cal_PearsonCorrelation(L148))
    print("CC: ", cal_cosineSimilarity(L148))
    # 计算相关性矩阵
    # correlation_matrix = row_correlation(L148[3], L148[4])

    # 打印相关性矩阵
    # print(correlation_matrix)
    # viewer_adj_single(correlation_matrix)

    # viewer_adj_single(L148[5])

