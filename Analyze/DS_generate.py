import os
import numpy as np
import random

# index:0-147，所以提取特殊点的特征的时候需要将其index-1
ds_node_path = r"D:\DataSet\DS_task_rest\task\DS_compare_fc_top20\node"
ds_edge_path = r"D:\DataSet\DS_task_rest\task\DS_compare_fc_top20\edge"

# 1.获取需要提取的节点index（并集）
# 获取需要提取的节点index（并集）
def get_union_index():
    from scipy import io

    # 读取.mat文件
    data = io.loadmat('all_top.mat')
    # 访问变量
    result = data['result']
    dataset = result[0][0]

    # 将所有的dataset提取出来存入列表（包括第一行静息态）
    dataset_list = []
    for (i, row) in enumerate(dataset):
        dataset_list.append(row)

    # 得到所有八类top30节点列表
    # print(dataset_list)

    # 取列表的并集
    dataset_index = []
    seen_elements = set()  # 创建一个空集合，用于存储已经出现的元素
    for list in dataset_list:
        for index in list:
            if index not in seen_elements:
                dataset_index.append(index)
                seen_elements.add(index)

    union_index = sorted(dataset_index)  # 排序
    print("######rest+task七类并集节点：" + f'{len(union_index)}')
    print(union_index)

    return union_index

# 2.把ds_generate处复制过去的特征文件全部修改成只有index指示的节点的邻接矩阵等再复写成仅有56 * 56表示的邻接矩阵
# 对node部分进行提取
def ds_node_generate(node_index):
    # 文件夹所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(ds_node_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(file_paths)
    for file_name in file_paths:
        # 读取原始文件的内容
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # 提取节点数和完整的功能连接矩阵
        node_count = int(lines[0])
        fc_matrix = []
        for line in lines[1:]:
            row = line.strip().split()
            row = [float(value) for value in row]
            fc_matrix.append(row)
        # print(fc_matrix)

        # 创建只包含指定节点的邻接矩阵
        new_fc_matrix = []
        for i in node_index:
            row = []
            for j in node_index:
                row.append(fc_matrix[i - 1][j - 1])     # -1已经考虑了序号对应问题了
            new_fc_matrix.append(row)
        #print(new_fc_matrix)
        # 更新节点数为指定节点的数量
        new_node_count = len(node_index)

        # 将节点数和新的邻接矩阵写入原始文件
        with open(file_name, 'w') as file:
            file.write("" + str(new_node_count) + "\n")
            for row in new_fc_matrix:
                file.write(" ".join(str(value) for value in row) + "\n")

    print("ds_node_generate.......Done!")

# 3.把提取的节点序号表转到1-56去。要不结构矩阵会出问题
def reverse_node_index(node_index):
    node_dictionary = {}
    for i, index in enumerate(node_index):
        node_dictionary[index] = i + 1
    print(node_dictionary)
    return node_dictionary

# 对edge部分进行提取
def ds_edge_generate(node_index):
    # 文件夹所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(ds_edge_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    # print(file_paths)
    node_index_dictionary = reverse_node_index(node_index)  # 把提取的节点序号表转到1-53去。要不结构矩阵会出问题
    for file_name in file_paths:
        # 读取原始文件的内容
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # 提取边的数量和完整的边列表
        edge_count = int(lines[0].strip())
        edges = [line.strip().split() for line in lines[1:]]
        edges = [(int(edge[0]), int(edge[1])) for edge in edges]

        # 筛选只包含指定节点的边
        filtered_edges = []
        for edge in edges:
            if edge[0] in node_index and edge[1] in node_index:
                filtered_edges.append(edge)

        # print(filtered_edges)      # 选出来的边，但为了变成一个53*53结果的把这个序号得对上
        # 更新边的数量为筛选后的边数
        new_edge_count = len(filtered_edges)

        # 将筛选后的边的数量和边列表写入原始文件
        with open(file_name, 'w') as file:
            file.write(str(new_edge_count) + "\n")
            for edge in filtered_edges:
                file.write(str(node_index_dictionary[edge[0]]) + " " + str(node_index_dictionary[edge[1]]) + "\n")
    print("ds_edge_generate.......Done!")

# 随机抽取Node_index以外的len(node_index）个节点序号，以利用以上函数生成新的数据集测试效果
def get_node_index_random(node_index):
    all_nodes = list(range(1, 149))  # 从1-148的节点序号
    remain_nodes = [node for node in all_nodes if node not in node_index]
    # print(remain_nodes)
    node_index_random = random.sample(remain_nodes, len(node_index))
    node_index_random.sort()
    print(node_index_random)
    return node_index_random

# 随机抽取不完全等同于Node_index的len(node_index）个节点序号，以利用以上函数生成新的数据集测试效果
def get_node_index_random2(node_index):
    all_nodes = list(range(1, 149))  # 从1-148的节点序号

    while True:
        node_index_random = random.sample(all_nodes, len(node_index))
        node_index_random.sort()

        if node_index_random != node_index:
            break

    coverage = len(set(node_index) & set(node_index_random)) / len(node_index) * 100   # 看新序列覆盖了先验的集合里的%多少
    print(coverage)

    return node_index_random, coverage


if __name__ == '__main__':
    # 1.提取出先验的
    # node_index = get_union_index()

    # 2.从除了先验以外的节点中随机先验个
    # node_index = get_node_index_random(node_index)

    # 3.从全体节点随机先验个，可重叠但不完全等同
    #node_index, coverage = get_node_index_random2(node_index)
    #print(node_index)
    #print(coverage)
    # node_index = [1, 3, 5, 9, 12, 17, 18, 23, 24, 31, 32, 33, 34, 35, 37, 39, 42, 43, 46, 47]  # 生成数据集节点特征部分

    # 七分类筛出新提取的56个节点
    # node_index = [2, 11, 19, 21, 22, 26, 27, 38, 42, 44, 46, 51, 56, 57, 58, 59, 60, 65, 67, 68, 69, 73, 76, 81, 85, 86, 88, 93, 94, 95, 96, 99, 100, 101, 103, 104, 110, 114, 116, 118, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 139, 141, 142, 143, 147]
    # [38, 42, 46, 51, 57, 59, 60, 76, 86, 88, 110, 114, 125, 129, 131, 133, 134, 135, 137, 143]  20个差异节点
    # node_index = get_node_index_random(node_index)
    # print(node_index)

    # 七类里的交集15个节点单独分类呢
    # node_index = [15, 16, 19, 27, 28, 29, 56, 67, 89, 90, 101, 103, 126, 128, 130]

    # 56里排序最前的20个节点
    # node_index = sorted([56, 2, 130, 21, 142, 116, 100, 68, 58, 132])
    # node_index = sorted([56, 130, 68, 21, 2, 100, 142, 116, 19, 132, 95, 58, 139, 65, 93, 89, 27, 99, 101, 96, 26, 69, 81, 128, 104, 44, 118, 52, 7, 67])

    # node_index = sorted([133, 59, 19, 27, 130, 58, 93, 132, 57, 131, 101, 56, 142, 147, 100, 67, 134, 2, 76, 125, 69, 51, 129, 60, 143, 88, 141, 95, 68, 104, 94, 21, 111, 136, 73, 112, 146, 86, 116, 65, 26, 128, 99, 120, 22, 135, 20, 108, 4, 90, 50, 127, 30, 85, 137, 11])
    # select LM top56

    # node_index = [1, 3, 6, 7, 10, 12, 14, 15, 16, 17, 18, 23, 24, 25, 29, 30, 34, 36, 39, 40, 41, 43, 45, 47, 52, 54, 55, 61, 64, 66, 74, 75, 79, 83, 84, 89, 91, 92, 98, 105, 106, 107, 108, 111, 112, 113, 117, 119, 120, 122, 123, 136, 140, 144, 145, 146]
    # node_index = random.sample(node_index, 20)
    # print(sorted(node_index))

    # node_index = [3, 10, 12, 15, 17, 23, 36, 39, 40, 43, 55, 83, 89, 91, 106, 111, 112, 136, 144, 145]  random20


    # node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52]
    # 连接性最高的non-overlapping hub + hub (就直接取前20个功能连接性最强的）(刚好10Common,10non-overlapping hub)
    node_index = [90, 16, 89, 15, 29, 103, 128, 127, 104, 30, 126, 26, 27, 67, 28, 101, 54, 19, 45, 130]
    print(len(node_index))
    ds_node_generate(node_index)
    # 生成数据集结构部分
    ds_edge_generate(node_index)
