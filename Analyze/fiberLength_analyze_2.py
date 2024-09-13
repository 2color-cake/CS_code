import scipy.io
import numpy as np
import matplotlib.pyplot as plt

node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]
node = [2, 11, 19, 21, 22, 26, 27, 38, 42, 44, 46, 51, 56, 57, 58, 59, 60, 65, 67, 68, 69, 73, 76, 81, 85, 86, 88, 93, 94, 95, 96, 99, 100, 101, 103, 104, 110, 114, 116, 118, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 139, 141, 142, 143, 147]
intersection = [2, 11, 19, 21, 22, 26, 27, 44, 56, 58, 65, 67, 68, 69, 73, 81, 85, 93, 94, 95, 96, 99, 100, 101, 103, 104, 116, 118, 127, 128, 130, 132, 139, 141, 142, 147]
scale1_non_intersection = [4, 6, 7, 15, 16, 20, 25, 28, 29, 30, 45, 52, 53, 54, 80, 89, 90, 102, 119, 126]
Select_non_intersection = [38, 42, 46, 51, 57, 59, 60, 76, 86, 88, 110, 114, 125, 129, 131, 133, 134, 135, 137, 143]
node_56inner20 = [130, 68, 56, 2, 21, 100, 142, 116, 58, 132, 19, 95, 93, 89, 139, 27, 101, 96, 65, 69]


# 读取MATLAB文件
mat = scipy.io.loadmat('114621_fiberLength.mat')

# 查看文件中的变量
# print(mat.keys())

# 访问特定变量 # 148 * 148只包含右上三角的各个区域连接的长度
LENSAtlas = np.array(mat['LENSAtlas'])

# print(LENSAtlas)

# LENSAtlas是每个点之间纤维长度(只有右上，小index->大index部分）
# 计算穿过单个区域的平均纤维长度
def cross_avg_fiberlength_single_node(node):
    rows, cols = LENSAtlas.shape
    length = []
    for i in range(rows):
        for j in range(cols):
            if i+1 == node and LENSAtlas[i][j] != 0:
                length.append(LENSAtlas[i][j])
            if j+1 == node and LENSAtlas[i][j] != 0:
                length.append(LENSAtlas[i][j])
    # print(length)
    sum_length = sum(length)
    avg_length = sum_length / len(length)   # 只统计穿过的，而不是看谁穿过的多
    return avg_length

def cross_avg_fiberlength_node_list(n_list):
    length = []
    for node in n_list:
        length.append(cross_avg_fiberlength_single_node(node))
    print(f"#####穿过节点{n_list}的平均纤维长度： \n", sum(length)/len(length))
    print(length)
    return sum(length)/len(length), length

def all_figure_colorful():
    # 计算各个节点类型的平均长度
    red_avg_length, _ = cross_avg_fiberlength_node_list(node_index)  # Scale-1平均
    pink_avg_length, _ = cross_avg_fiberlength_node_list(intersection) # common
    orange_avg_length, _ = cross_avg_fiberlength_node_list(scale1_non_intersection)  #non-overlapping HUb
    yellow_avg_length, _ = cross_avg_fiberlength_node_list(Select_non_intersection)  #Specific
    green_avg_length, _ = cross_avg_fiberlength_node_list(node)  # 前56平均           #KBRs
    avg_all, _ = cross_avg_fiberlength_node_list([i + 1 for i in range(148)])

    # 设置数据和标签
    labels = ['Scale-1', 'Intersection', 'No Intersect\nin Scale-1', 'No Intersect\nin Select', 'Select 56', 'All']
    avg_lengths = [red_avg_length, pink_avg_length, orange_avg_length, yellow_avg_length, green_avg_length, avg_all]

    # 绘制柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(labels, avg_lengths, edgecolor="blue",
            color=['tomato', 'mistyrose', 'gold', 'skyblue', 'slateblue', 'yellowgreen'], width=0.6)

    # 添加标签和标题
    plt.xlabel('Node Type')
    plt.ylabel('Average Fiber Length')
    plt.title('Average Fiber Length for Different Node Types')

    # 显示图形
    plt.show()

def box_line_figure():
    # 计算各个节点类型的纤维长度列表
    print("hub")
    red_avg_length, scale_1 = cross_avg_fiberlength_node_list(node_index)  # Scale-1平均
    print("common")
    pink_avg_length, common_nodes = cross_avg_fiberlength_node_list(intersection)
    print("non-overlapping hub")
    orange_avg_length, scale_1_unique = cross_avg_fiberlength_node_list(scale1_non_intersection)
    print("specific")
    yellow_avg_length, selected_unique = cross_avg_fiberlength_node_list(Select_non_intersection)
    green_avg_length, selected = cross_avg_fiberlength_node_list(node)  # 前56平均
    avg_all, all = cross_avg_fiberlength_node_list([i + 1 for i in range(148)])
    _, _ = cross_avg_fiberlength_node_list(node_56inner20)

    '''
    all = np.array(all)
    # 按照从高到低排序的列号
    sorted_columns = np.argsort(-all)  # 使用负号实现从高到低排序
    sorted_columns += 1
    # 输出排序后的列号
    print("Graph_sum_weight_rank : \n", sorted_columns)
    sorted_columns = sorted_columns[:56]
    print(sorted_columns)
    print("与scale-1交集", len(set(sorted_columns).intersection(set(node_index))))
    print("与selected交集", len(set(sorted_columns).intersection(set(node))))
    print("与common交集", len(set(sorted_columns).intersection(set(intersection))))
    '''

   #  # 节点类型和对应的平均纤维长度
   #  data = [scale_1, selected, common_nodes, scale_1_unique, selected_unique,  all]  # scale_1_unique, selected_unique,
   #  labels = ['Hub', 'KBRs', 'Common', 'Non-overlapping Hub', 'Specific', 'All']  # 'Scale-1 Unique', 'Select Unique',
   # # 节点类型和对应的平均纤维长度
    data = [scale_1, selected, all]  # scale_1_unique, selected_unique,
    labels = ['Hub', 'KBRs', 'All']  # 'Scale-1 Unique', 'Select Unique',
    # 绘制箱线图比较该五个
    # 绘制箱线图
    plt.figure(dpi=600)
    plt.boxplot(data, labels=labels)

    # 添加标题和标签
    plt.title('Fiber Length Distribution')
    # plt.xlabel('Node')
    plt.ylabel('Fiber Length')

    # 显示图形
    plt.tight_layout()
    plt.show()

box_line_figure()

