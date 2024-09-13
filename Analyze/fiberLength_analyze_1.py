import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# 读取MATLAB文件
mat = scipy.io.loadmat('114621_fiberLength.mat')

# 查看文件中的变量
# print(mat.keys())

# 访问特定变量 # 148 * 148只包含右上三角的各个区域连接的长度
LENSAtlas = np.matrix(mat['LENSAtlas'])

print(LENSAtlas.shape)


# red
node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]
newSelected = [2, 11, 19, 21, 22, 26, 27, 38, 42, 44, 46, 51, 56, 57, 58, 59, 60, 65, 67, 68, 69, 73, 76, 81, 85, 86, 88, 93, 94, 95, 96, 99, 100, 101, 103, 104, 110, 114, 116, 118, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 139, 141, 142, 143, 147]
# pink
intersection_nodes = [2, 11, 19, 21, 22, 26, 27, 44, 56, 58, 65, 67, 68, 69, 73, 81, 85, 93, 94, 95, 96, 99, 100, 101, 103, 104, 116, 118, 127, 128, 130, 132, 139, 141, 142, 147]
# yellow
non_intersection_nodes = [38, 42, 46, 51, 57, 59, 60, 76, 86, 88, 110, 114, 125, 129, 131, 133, 134, 135, 137, 143]
non_intersection_Scale1 = [4, 6, 7, 15, 16, 20, 25, 28, 29, 30, 45, 52, 53, 54, 80, 89, 90, 102, 119, 126]
# 计算平均FiberLength
def cal_avg_fiber_length(node_indices, LENSAtlas):
    total_length = 0
    count = 0
    for i in range(len(node_indices)):
        for j in range(len(node_indices)):
            node_i = node_indices[i] - 1  # 从1-based转换为0-based
            node_j = node_indices[j] - 1  # 从1-based转换为0-based
            if node_i != node_j:
                total_length += LENSAtlas[node_i, node_j]
                count += 1
    if count > 0:
        avg_length = total_length / count
        return avg_length * 2   # 上三角矩阵，这个多加了一倍的0
    else:
        return None

avg_all = cal_avg_fiber_length([x for x in range(1, 149)], LENSAtlas)
print("###All Nodes Average Fiber Length:", avg_all)


# 1.计算Scale-1节点之间的平均纤维长度
red_avg_length = cal_avg_fiber_length(node_index, LENSAtlas)
print("###Scale-1 Nodes Average Fiber Length:", red_avg_length)

# 2.计算Scale-1和七分类前56交集节点之间的平均纤维长度
pink_avg_length = cal_avg_fiber_length(intersection_nodes, LENSAtlas)
print("###Intersection Nodes Average Fiber Length:", pink_avg_length)

# 3.计算Scale-1和七分类前56非交集节点之间的平均纤维长度
yellow_avg_length = cal_avg_fiber_length(non_intersection_nodes, LENSAtlas)
print("###Non Intersection Nodes Average Fiber Length:", yellow_avg_length)

# 4.计算黄色+粉色节点之间的平均纤维长度
# newSelected = [130, 57, 51, 56, 116, 131, 27, 125, 42, 76, 2, 59, 58, 133, 132, 19, 21, 95, 93, 142, 101, 147, 68, 100, 135, 73, 67, 129, 69, 88, 134, 85, 65, 96, 139, 22, 86, 143, 60, 137, 127, 11, 118, 141, 114, 99, 110, 104, 128, 81, 44, 20, 7, 89, 26, 120]
newSelected_avg_length = cal_avg_fiber_length(newSelected, LENSAtlas)
print("###Selected Nodes Average Fiber Length:", newSelected_avg_length)

# 非交集部分的Scale-1 比其他的长很多、、
newNonIntersection_length = cal_avg_fiber_length(non_intersection_Scale1, LENSAtlas)
print("###newNonIntersection Nodes Average Fiber Length:", newNonIntersection_length)


# 绘制直方图，并设置图形大小
plt.figure(figsize=(10, 6))  # 设置图形大小为宽度8英寸，高度6英寸

# 设置数据
labels = ['Scale-1', 'Intersection', 'Non Intersect\nfor Scale-1', 'Non Intersect\nfor Selected', 'Selected 56', 'All']
avg_lengths = [red_avg_length, pink_avg_length, newNonIntersection_length, yellow_avg_length, newSelected_avg_length, avg_all]

# 绘制柱状图，设置柱子宽度
plt.bar(labels, avg_lengths, color=['red', 'pink', 'red', 'yellow', 'orange', 'skyblue'], width=0.2)

# 添加标签和标题
plt.xlabel('Node Type')
plt.ylabel('Average Fiber Length')
plt.title('Average Fiber Length for Different Node Types')
# plt.xticks(rotation=60)
plt.tight_layout()

# 显示图形
plt.show()
