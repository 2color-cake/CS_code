import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
        L.append(Min_Max_Norm(np.matrix(L_data)))    # 归一化后的
        # io.savemat(f"{file_name}.mat", {'array': L_data})

    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            L[i] = np.array(L[i])  # 原本是matrix类型对象，转成ndarray形式
            L[j] = np.array(L[j])
    return L

# 绘制矩阵
def draw_matrix(matrix):
    # 设置颜色条的范围
    vmin = 0
    vmax = 0.3
    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'])
    # 绘制矩阵
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()  # 添加颜色条
    plt.title("7-2")
    plt.show(dpi=600)

# 绘制三个指标对比图
def draw_SSIM_PC_CC():
    import matplotlib.pyplot as plt

    # 任务1的平均指标值
    ssim_single = 0.18271321836944976
    pearson_single = 0.04190479865523069
    cosine_single = 0.1307892072583211

    # 任务2的平均指标值
    ssim_sum = 0.541732341896401
    pearson_sum = 0.5859256090744294
    cosine_sum = 0.8025685884270864

    # 指标名称
    # indicators = ['SSIM', 'Pearson Correlation', 'Cosine Correlation']
    indicators = ['SSIM', 'Cosine Similarity']

    # 指标值
    # task1_values = [ssim_single, pearson_single, cosine_single]
    # task2_values = [ssim_sum, pearson_sum, cosine_sum]
    task1_values = [ssim_single, cosine_single]
    task2_values = [ssim_sum, cosine_sum]

    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    # 比较任务1和任务2的指标值
    width = 0.35  # 条形的宽度
    x = range(len(indicators))

    bar1 = ax.bar(x, task1_values, width, label='Single', color="royalblue")  #edgecolor="b"
    # bar2 = ax.bar([i + width for i in x], task2_values, width, label='Sum-15', color="salmon")
    bar2 = ax.bar([i + width for i in x], task2_values, width, label='Sum-10', color="salmon")


    # 添加指标值标签
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    # 设置标题和标签
    # ax.set_xlabel('Indicators', fontsize=14)
    ax.set_ylabel('Values', fontsize=14)
    # ax.set_title('Comparison of Average Indicators between Single and Sum-15', fontsize=14)
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(indicators, fontsize=14)
    ax.legend(fontsize=14)
    # 设置y轴刻度间隔
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", linestyle='--', linewidth=0.5)

    # 显示图形
    plt.show()

# 绘制六个任务的相似度图
def draw_CC_6tasks():
    import matplotlib.pyplot as plt
    import itertools

    # CC值
    cc_values = [
        0.8961252702865649, 0.8754963556619646, 0.83775703628521,
        0.8318535155978646, 0.808672306683419, 0.8686199505674049,
        0.8287747632361443, 0.8257229834400047, 0.8090924028934205,
        0.8115686521892973, 0.8183465272696617, 0.800560295091173,
        0.7821027561940893, 0.7664285243560642, 0.774846990561456
    ]

    # 生成两两组合的任务标签
    tasks_combinations = list(itertools.combinations(range(6), 2))

    # 生成两两组合的任务对标签
    task_labels = [f'{i + 2}-{j + 2}' for i, j in tasks_combinations]

    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

    # 绘制CC值折线图
    ax.plot(task_labels, cc_values, color="royalblue", marker='o', label='Cosine Similarity between Adjacency Matrixs')

    # 绘制平均CC值水平线
    average_cc = sum(cc_values) / len(cc_values)
    ax.axhline(y=average_cc, color='tomato', linestyle='--', label=f'Average: {average_cc:.4f}')

    # 添加标题和标签
    ax.set_xlabel('Task Combination', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('Cosine Similarity between Adjacency Matrixs', fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis="y", linestyle='--', linewidth=0.5)

    # 自动调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

# 六个任务的区域交集 IoU
def draw_nodes_intersection_6tasks():
    import matplotlib.pyplot as plt

    # 任务组合的序号
    task_combinations = [
        '2-3', '2-4', '2-5', '2-6', '2-7',
        '3-4', '3-5', '3-6', '3-7',
        '4-5', '4-6', '4-7',
        '5-6', '5-7',
        '6-7'
    ]

    # 节点交集数量
    intersection_counts = [
        44, 42, 42, 40, 43,
        44, 42, 44, 46,
        43, 43, 46,
        44, 43,
        51
    ]
    # IoU 交集比例
    IoU = []   # inter of union
    for inter in intersection_counts:
        IoU.append(inter/(2*(56-inter) + inter))

    # 重要节点数量
    important_node_counts = [
        27, 23, 22, 23, 25,
        27, 26, 28, 30,
        24, 26, 28,
        24, 26,
        34
    ]
    # 计算平均交集数量
    average_intersection = np.mean(intersection_counts)

    # 计算平均IoU值
    average_IoU = np.mean(IoU)

    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

    # 绘制节点交集数量条形图
    x = range(len(task_combinations))
    bars = ax.bar(x, IoU, width=0.4, label='Intersection of Union', color='skyblue')

    # 添加平均交集数量的直线
    ax.axhline(y=average_IoU, color='red', linestyle='--', label=f'Average: {0.644437}')

    # 添加标题和标签
    ax.set_xlabel('Task Combination', fontsize=14)
    ax.set_ylabel('IoU', fontsize=14)
    ax.set_title('Comparison of Intersection of Union between Task Combinations', fontsize=14)

    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(task_combinations)

    # 添加图例
    ax.legend(fontsize=14)
    # 添加背景线
    ax.grid(axis="y", linestyle='--', linewidth=0.5)
    # 显示图形
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
'''
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制节点交集数量条形图
    x = range(len(task_combinations))
    ax.bar(x, intersection_counts, width=0.4, label='Top56 Nodes Intersection Counts', color='tomato')

    # 绘制重要节点数量折线图
    ax.plot(x, important_node_counts, color='blue', linestyle='-', marker='o', label='Important Node Counts')

    # 添加标题和标签
    ax.set_xlabel('Task Combinations')
    ax.set_ylabel('Counts')
    ax.set_title('Comparison of Intersection Counts between Task Combinations')

    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(task_combinations)

    # 添加图例
    ax.legend()

    # 显示图形
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
'''

# data = np.loadtxt("L_init_xavier_normal.txt")
# draw_matrix(data)
# data = np.loadtxt(".\data\\7-L148\L10.txt")
# draw_matrix(data)

# 导入邻接矩阵列表（30个）
# L148_path = r"D:\Projects\model\DS_generate-rest_task\data\7-L56"   # 120个图稳定性多判断几个
# L148 = load_Ldata(L148_path)
# tmp = np.sum(L148[10:25], axis=0, dtype=np.float64)
# tmp = Min_Max_Norm(tmp)
# draw_matrix(tmp)

# draw_SSIM_PC_CC()
draw_CC_6tasks()
# draw_nodes_intersection_6tasks()