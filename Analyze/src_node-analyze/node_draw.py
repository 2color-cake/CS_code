import matplotlib.pyplot as plt
import numpy as np

# 148*148边
def node_7_148():
    # 数据
    percentage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 百分比
    total_nodes = [15, 30, 45, 60, 75, 90, 105, 120, 135, 148]  # 阶段性总节点数
    internal_nodes = [8, 19, 30, 40, 45, 50, 50, 55, 56, 56]  # 阶段性内部节点数

    # 计算阶段性外部节点数
    external_nodes = [total - internal for total, internal in zip(total_nodes, internal_nodes)]

    # 绘制堆叠柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(percentage, np.array(external_nodes) / total_nodes, bottom=np.array(internal_nodes) / total_nodes,
            label='Others', edgecolor="blue", color='grey', width=5)
    plt.bar(percentage, np.array(internal_nodes) / total_nodes, edgecolor="black", label='Scale-1 Nodes', color='cornflowerblue', width=5)

    # 添加标签和标题
    plt.xlabel('Percentage')
    plt.ylabel('Ratio of Nodes')
    plt.title('Scale-1 Nodes Ratio for Different Percentage')
    plt.xticks(percentage)
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()

node_7_148()