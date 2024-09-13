import matplotlib.pyplot as plt
import numpy as np

# 148*148边
def edge_7_148():
    # 数据
    percentage = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # 百分比
    all_edges = np.array([1079, 2158, 3238, 4317, 5397, 6476, 7555, 8635, 9714, 10794])  # 总边数
    both_edges = np.array([204, 422, 611, 784, 969, 1114, 1263, 1400, 1497, 1577])  # 两点均为node_index中节点的边数
    single_edges = np.array([558, 1102, 1662, 2236, 2755, 3272, 3779, 4245, 4683, 5074])  # 只有一点是node_index中节点的边数
    none_edges = np.array([317, 634, 965, 1297, 1673, 2090, 2513, 2990, 3534, 4143])  # 两点都不是node_index中节点的边数

    # 计算每种边的比例
    both_ratios = both_edges / all_edges * 100
    single_ratios = single_edges / all_edges * 100
    none_ratios = none_edges / all_edges * 100

    # 绘制堆叠柱形图
    plt.figure(figsize=(10, 6))
    plt.bar(percentage, none_ratios, edgecolor="blue", label='None Node in Scale-1', color='moccasin', width=5)
    plt.bar(percentage, single_ratios, edgecolor="blue", bottom=none_ratios, label='One Node in Scale-1', color='yellowgreen', width=5)
    plt.bar(percentage, both_ratios, edgecolor="blue", bottom=none_ratios + single_ratios, label='Both Nodes in Scale-1'
            , color='royalblue', width=5)

    # 添加标签和标题
    plt.xlabel('Percentage of Edges')
    plt.ylabel('Edge Ratio (%)')
    plt.title('Edge Ratio for Different Types of Edges')
    plt.xticks(percentage)
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()

edge_7_148()