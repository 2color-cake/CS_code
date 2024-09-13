import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import math
import networkx as nx

# influential nodes:
node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]

# rank: ES  ELS  ELSW  ELMSW  ELMSWR  ELMSWRG
node2 = [133, 59, 101, 27, 69, 100, 131, 147, 19, 67, 57, 93, 130, 132, 142, 56, 143, 141, 88, 104, 58, 2, 125, 60, 134, 128, 112, 68, 76, 116, 26, 110, 137, 114, 111, 129, 51, 120, 87, 73, 94, 95, 90, 32, 102, 42, 124, 85, 11, 46, 103, 49, 144, 99, 105, 86, 21, 127, 30, 106, 121, 7, 146, 113, 63, 89, 28, 50, 135, 145, 20, 29, 126, 123, 61, 71, 5, 22, 139, 10, 31, 37, 40, 119, 140, 138, 36, 54, 14, 91, 65, 108, 84, 78, 79, 118, 82, 45, 107, 122, 39, 80, 44, 72, 62, 23, 83, 52, 38, 3, 48, 4, 96, 12, 15, 98, 115, 66, 13, 43, 8, 74, 53, 24, 81, 109, 9, 70, 92, 77, 75, 97, 25, 16, 34, 1, 136, 148, 47, 6, 41, 35, 17, 64, 55, 33, 18, 117]
node3 = [133, 59, 19, 27, 130, 58, 93, 132, 57, 131, 101, 56, 142, 147, 100, 67, 134, 2, 76, 125, 69, 51, 129, 60, 143, 88, 141, 95, 68, 104, 94, 21, 111, 136, 73, 112, 146, 86, 116, 65, 26, 128, 99, 120, 22, 135, 20, 108, 4, 90, 50, 127, 30, 85, 137, 11, 103, 110, 42, 139, 87, 61, 53, 34, 102, 89, 121, 79, 96, 82, 123, 145, 144, 114, 49, 35, 124, 126, 7, 14, 10, 43, 32, 105, 75, 5, 106, 13, 117, 45, 36, 17, 140, 1, 15, 119, 72, 92, 28, 33, 46, 66, 29, 48, 54, 64, 78, 84, 98, 62, 83, 8, 109, 77, 115, 70, 38, 71, 3, 97, 113, 18, 37, 44, 6, 31, 63, 9, 122, 80, 118, 81, 91, 23, 24, 52, 41, 16, 138, 47, 40, 25, 39, 148, 12, 74, 55, 107]
node4 = [19, 59, 57, 133, 131, 130, 93, 58, 132, 27, 56, 51, 101, 125, 2, 116, 147, 21, 42, 76, 100, 69, 142, 73, 60, 88, 95, 67, 129, 134, 135, 50, 68, 120, 137, 143, 20, 141, 38, 114, 61, 81, 65, 128, 29, 124, 4, 32, 86, 139, 94, 11, 85, 106, 89, 23, 34, 110, 104, 123, 96, 7, 127, 102, 43, 49, 22, 112, 35, 26, 108, 99, 1, 146, 14, 62, 53, 46, 13, 119, 145, 28, 78, 17, 84, 31, 126, 121, 39, 48, 70, 144, 72, 118, 52, 138, 136, 103, 45, 105, 109, 10, 87, 91, 111, 36, 64, 40, 15, 5, 71, 90, 24, 115, 117, 77, 30, 47, 54, 83, 37, 12, 82, 98, 63, 113, 140, 75, 97, 122, 3, 33, 107, 25, 79, 6, 9, 92, 66, 44, 74, 18, 148, 41, 80, 16, 8, 55]
node5 = [57, 131, 59, 133, 51, 116, 19, 2, 130, 93, 125, 76, 132, 58, 42, 21, 147, 27, 56, 73, 134, 142, 88, 120, 100, 137, 65, 95, 60, 101, 69, 86, 135, 67, 138, 20, 105, 128, 129, 68, 87, 77, 81, 127, 91, 143, 90, 141, 124, 34, 26, 53, 31, 49, 106, 15, 43, 139, 89, 94, 3, 119, 13, 99, 70, 11, 102, 22, 108, 82, 103, 98, 110, 121, 1, 35, 38, 28, 50, 4, 7, 14, 46, 145, 115, 104, 29, 112, 85, 44, 62, 78, 32, 123, 52, 96, 114, 54, 113, 40, 84, 48, 36, 61, 30, 6, 144, 75, 97, 23, 136, 10, 17, 126, 117, 45, 18, 118, 79, 74, 41, 111, 146, 8, 83, 5, 107, 80, 25, 12, 47, 122, 16, 39, 148, 71, 140, 24, 55, 66, 33, 92, 63, 72, 109, 9, 37, 64]
node6 = [131, 57, 51, 59, 130, 132, 116, 133, 2, 19, 42, 125, 93, 56, 76, 27, 58, 147, 21, 100, 68, 95, 73, 142, 101, 134, 69, 88, 135, 129, 65, 22, 67, 137, 60, 120, 85, 104, 26, 7, 138, 86, 44, 96, 141, 143, 20, 139, 105, 118, 89, 127, 126, 11, 52, 31, 50, 128, 98, 49, 110, 106, 99, 90, 81, 77, 87, 102, 94, 114, 34, 145, 53, 70, 14, 84, 75, 108, 43, 36, 91, 30, 124, 38, 15, 62, 103, 46, 136, 78, 112, 54, 1, 35, 140, 123, 25, 144, 32, 12, 23, 37, 61, 115, 119, 92, 29, 10, 121, 45, 3, 40, 72, 146, 113, 148, 4, 24, 79, 13, 6, 16, 63, 71, 80, 82, 107, 5, 111, 122, 9, 18, 109, 17, 83, 28, 74, 39, 41, 66, 48, 117, 55, 33, 64, 47, 97, 8]
node7 = [130, 57, 51, 56, 116, 131, 27, 125, 42, 76, 2, 59, 58, 133, 132, 19, 21, 95, 93, 142, 101, 147, 68, 100, 135, 73, 67, 129, 69, 88, 134, 85, 65, 96, 139, 22, 86, 143, 60, 137, 127, 11, 118, 141, 114, 99, 110, 104, 128, 81, 44, 20, 7, 89, 26, 120, 46, 102, 43, 38, 103, 94, 52, 77, 25, 106, 108, 145, 14, 30, 123, 112, 15, 138, 144, 84, 40, 124, 31, 105, 50, 29, 90, 3, 115, 54, 121, 146, 78, 62, 34, 136, 32, 126, 49, 61, 53, 12, 91, 70, 36, 13, 37, 119, 87, 111, 6, 72, 1, 122, 28, 98, 4, 24, 23, 79, 117, 109, 16, 35, 66, 82, 75, 92, 97, 10, 80, 83, 71, 113, 63, 148, 45, 55, 8, 5, 140, 47, 107, 48, 64, 17, 9, 18, 39, 74, 33, 41]

# 1.交集操作(前%多少)
def cal_intersection(percentage):
    # 设置每个任务认为最重要的节点数量的百分比（例如前10%）
    percentage_top_nodes = percentage

    # 计算每个任务取前几个节点的数量
    num_top_nodes = int(len(node2) * percentage_top_nodes)

    # 取每个任务认为最重要的前几个节点
    top_nodes_task2 = set(node2[:num_top_nodes])
    top_nodes_task3 = set(node3[:num_top_nodes])
    top_nodes_task4 = set(node4[:num_top_nodes])
    top_nodes_task5 = set(node5[:num_top_nodes])
    top_nodes_task6 = set(node6[:num_top_nodes])
    top_nodes_task7 = set(node7[:num_top_nodes])

    # 取所有任务中共同的最重要节点
    print(f"#####六个任务中前{percentage_top_nodes * 100}%的节点交集：")
    common_top_nodes = set.intersection(top_nodes_task2, top_nodes_task3, top_nodes_task4, top_nodes_task5,
                                        top_nodes_task6, top_nodes_task7)

    # print(sorted(common_top_nodes))
    print(common_top_nodes)

    print("#####其中重要节点：")
    influential_nodes = set(node_index)
    res = set.intersection(influential_nodes, common_top_nodes)
    # print(sorted(res))
    print(res)

# 2.交集操作(前多少个)
def get_topnum(num):
    num_top_nodes = num
    # 取每个任务认为最重要的前几个节点
    top_nodes_task2 = node2[:num_top_nodes]
    top_nodes_task3 = node3[:num_top_nodes]
    top_nodes_task4 = node4[:num_top_nodes]
    top_nodes_task5 = node5[:num_top_nodes]
    top_nodes_task6 = node6[:num_top_nodes]
    top_nodes_task7 = node7[:num_top_nodes]

    # print("# 2:\n", sorted(top_nodes_task2))
    # print("# 3:\n", sorted(top_nodes_task3))
    # print("# 4:\n", sorted(top_nodes_task4))
    # print("# 5:\n", sorted(top_nodes_task5))
    # print("# 6:\n", sorted(top_nodes_task6))
    # print("# 7:\n", sorted(top_nodes_task7))

    top_nodes_task2 = set(top_nodes_task2)
    top_nodes_task3 = set(top_nodes_task3)
    top_nodes_task4 = set(top_nodes_task4)
    top_nodes_task5 = set(top_nodes_task5)
    top_nodes_task6 = set(top_nodes_task6)
    top_nodes_task7 = set(top_nodes_task7)

    # 取所有任务中共同的最重要节点
    print(f"#####六个任务中前{num}个节点的交集：")
    common_top_nodes = set.intersection(top_nodes_task2, top_nodes_task3, top_nodes_task4, top_nodes_task5,
                                        top_nodes_task6, top_nodes_task7)

    # print(sorted(common_top_nodes))
    print(common_top_nodes)


    print("#####其中重要节点：")
    influential_nodes = set(node_index)
    res = set.intersection(influential_nodes, common_top_nodes)
    # print(sorted(res))
    print(res)

    tasks_interset = []
    tasks_interset_scale_1 = []

    # # 所有任务之间前num个点的交集，以及其中重要节点
    # for i in range(2, 8):
    #     for j in range(i + 1, 8):
    #         # 获取任务i和任务j的节点列表
    #         nodes_i = globals()[f'node{i}'][:num_top_nodes]
    #         nodes_j = globals()[f'node{j}'][:num_top_nodes]
    #         # 转换为集合
    #         nodes_i_set = set(nodes_i)
    #         nodes_j_set = set(nodes_j)
    #         # 计算节点列表的交集
    #         intersect_2tasks = set.intersection(nodes_i_set, nodes_j_set)
    #         print(f"###任务{i}和任务{j}前{num}个节点的交集：")
    #         print(len(intersect_2tasks))
    #         tasks_interset.append(len(intersect_2tasks))
    #         # 计算交集中的重要节点
    #         res = set.intersection(influential_nodes, intersect_2tasks)
    #         print(f"#任务{i}和任务{j}的交集中重要节点：")
    #         print(len(res))
    #         tasks_interset_scale_1.append(len(res))
    #
    # print(tasks_interset)
    # print(tasks_interset_scale_1)

    # 所有任务之间前num个点的交集，以及其中重要节点
    for i in range(2, 8):
        # 获取任务i和任务j的节点列表
        nodes_i = globals()[f'node{i}'][:num_top_nodes]
        # 转换为集合
        nodes_i_set = set(nodes_i)

        # 计算节点列表的交集
        intersect_2tasks = set.intersection(nodes_i_set, node_index)
        print(f"###任务{i}和scale-1节点的交集：")
        print(len(intersect_2tasks))


if __name__ == '__main__':
    # node = []     # 用于给没有,分隔的列表转成列表
    # node_2 = input().split(" ")
    # for i in node_2:
    #     if i == " " or i == "[" or i == "]" or i == '' or i == "\t":
    #         continue
    #     else:
    #         node.append(int(i))
    # print(node)

    # #[
    # cal_intersection(0.1)
    # cal_intersection(0.2)
    # cal_intersection(0.3)
    num_top_nodes = 56
    get_topnum(num_top_nodes)

    top_nodes_task7 = node7[:num_top_nodes]  # 前56个点
    print(node_index)
    print(sorted(top_nodes_task7))
    print(len(set(top_nodes_task7).intersection(set(node_index))))
    print(sorted(set(top_nodes_task7).intersection(set(node_index))))
    print(set(top_nodes_task7) - set(top_nodes_task7).intersection(set(node_index)))
    # 差集和fc做了交集
    # print((set(top_nodes_task7) - set(top_nodes_task7).intersection(set(node_index))).intersection(set([1, 129, 3, 131, 133, 8, 12, 143, 146, 24, 36, 37, 38, 41, 42, 46, 49, 57, 61, 64, 72, 76, 78, 79, 82, 86, 88, 108, 110, 111, 112, 120, 123])))

