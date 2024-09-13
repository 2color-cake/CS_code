### 按照vote方式排序重要节点序号，如7类都出现则设置为最高分

import os
import numpy as np
import random
import operator

# 覆盖率为0时随机
node_index_cov0 = [1, 3, 12, 13, 14, 18, 31, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 46, 49, 51, 59, 60, 62, 63,
                          66, 71, 74, 75, 78, 79, 82, 83, 84, 86, 91, 92, 97, 98, 106, 107, 108, 109, 110, 111, 115,
                          122, 123, 131, 133, 134, 138, 145, 146]

# 获取需要提取的节点index（并集）
def get_union_index():
    from scipy import io

    # 读取.mat文件
    data = io.loadmat('all_top.mat')
    # 访问变量
    result = data['result']
    dataset = result[0][0]

    # 将除了第一行的dataset提取出来存入列表
    dataset_list = []
    for (i, row) in enumerate(dataset):
        # if i != 0:
        dataset_list.append(row)

    # 得到0-6的类别top30节点列表
    # print(dataset_list)

    # 取0-6类别top30列表的并集
    dataset_index = []
    seen_elements = set()  # 创建一个空集合，用于存储已经出现的元素
    for list in dataset_list:
        for index in list:
            if index not in seen_elements:
                dataset_index.append(index)
                seen_elements.add(index)

    union_index = sorted(dataset_index)  # 排序
    print("######七类并集节点：" + f'{len(union_index)}')
    print(union_index)

    return union_index

# 获取按vote打分的节点字典
def get_important_vote(union_index):
    node_important_vote = {}
    for i in union_index:
        node_important_vote[i] = 0
    # print(node_important_vote)
    from scipy import io

    # 读取.mat文件
    data = io.loadmat('all_top.mat')
    # 访问变量
    result = data['result']
    dataset = result[0][0]

    # 将除了第一行的dataset提取出来存入列表
    dataset_list = []
    for (i, row) in enumerate(dataset):
        # if i != 0:
        dataset_list.append(row)
    # print(dataset_list)

    # 用vote机制给分，字典存储
    for i in range(len(dataset_list)):
        for j in range(len(dataset_list[i])):
            node_important_vote[dataset_list[i][j]] += 1


    node_important_vote = sorted(node_important_vote.items(), key=operator.itemgetter(1), reverse=True)
    print("######" + f"{node_important_vote}")
    return node_important_vote

# 把提取的节点序号表转到1-53去。要不结构矩阵会出问题
def reverse_node_index(node_index):
    node_dictionary = {}
    for i, index in enumerate(node_index):
        node_dictionary[index] = i + 1
    # print(node_dictionary)
    return node_dictionary

if __name__ == '__main__':
    union_index = get_union_index()
    node_important_dictionary = get_important_vote(union_index)  # 但里面好像是元组并非字典、
    node_dictionary = reverse_node_index(union_index)
    end = []
    for tup in node_important_dictionary:
        end.append(tuple([node_dictionary[tup[0]], tup[1]]))
    print(end)
'''
[(15, 7), (16, 7), (19, 7), (27, 7), (28, 7), (29, 7), (56, 7), (67, 7), (89, 7), (90, 7), (101, 7), (103, 7), (126, 7), (128, 7), (130, 7), (26, 6), (104, 6), (127, 6), (11, 5), (69, 5), (94, 5), (20, 4), (45, 4), (52, 4), (53, 4), (93, 4), (142, 4), (147, 4), (7, 3), (30, 3), (58, 3), (85, 3), (99, 3), (100, 3), (22, 2), (25, 2), (54, 2), (81, 2), (102, 2), (132, 2), (141, 2), (2, 1), (6, 1), (21, 1), (44, 1), (65, 1), (68, 1), (73, 1), (80, 1), (95, 1), (116, 1), (119, 1), (139, 1)]

[(5, 7), (6, 7), (7, 7), (13, 7), (14, 7), (15, 7), (22, 7), (25, 7), (32, 7), (33, 7), (39, 7), (41, 7), (45, 7), (47, 7), (48, 7), (12, 6), (42, 6), (46, 6), (4, 5), (27, 5), (35, 5), (8, 4), (18, 4), (19, 4), (20, 4), (34, 4), (52, 4), (53, 4), (3, 3), (16, 3), (23, 3), (31, 3), (37, 3), (38, 3), (10, 2), (11, 2), (21, 2), (30, 2), (40, 2), (49, 2), (51, 2), (1, 1), (2, 1), (9, 1), (17, 1), (24, 1), (26, 1), (28, 1), (29, 1), (36, 1), (43, 1), (44, 1), (50, 1)]
'''
