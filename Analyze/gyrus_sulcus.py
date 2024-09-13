# G:1  S:2   G&S:3 none:4
dic = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 2, 40: 2, 41: 2, 42: 4, 43: 4, 44: 2, 45: 2, 46: 2, 47: 2, 48: 2, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2, 64: 2, 65: 2, 66: 2, 67: 2, 68: 2, 69: 2, 70: 2, 71: 2, 72: 2, 73: 2, 74: 2, 75: 3, 76: 3, 77: 3, 78: 3, 79: 3, 80: 3, 81: 3, 82: 3, 83: 1, 84: 1, 85: 1, 86: 1, 87: 1, 88: 1, 89: 1, 90: 1, 91: 1, 92: 1, 93: 1, 94: 1, 95: 1, 96: 1, 97: 1, 98: 1, 99: 1, 100: 1, 101: 1, 102: 1, 103: 1, 104: 1, 105: 1, 106: 1, 107: 1, 108: 1, 109: 1, 110: 1, 111: 1, 112: 1, 113: 2, 114: 2, 115: 2, 116: 4, 117: 4, 118: 2, 119: 2, 120: 2, 121: 2, 122: 2, 123: 2, 124: 2, 125: 2, 126: 2, 127: 2, 128: 2, 129: 2, 130: 2, 131: 2, 132: 2, 133: 2, 134: 2, 135: 2, 136: 2, 137: 2, 138: 2, 139: 2, 140: 2, 141: 2, 142: 2, 143: 2, 144: 2, 145: 2, 146: 2, 147: 2, 148: 2}
# 60个G，16个G&S，68个S,4nONE
def get_dic():
    # G:1  S:2   G&S:3 none:4
    dic = {}
    # G&S
    for i in range(1, 9):
        dic[i] = 3
        dic[i + 74] = 3
    # G
    for i in range(9, 39):
        dic[i] = 1
        dic[i + 74] = 1

    # None
    for i in range(39, 44):
        dic[i] = 4
        dic[i + 74] = 4

    # 勉强
    for i in range(39, 42):
        dic[i] = 2
        dic[i + 74] = 2

    for i in range(44, 75):
        dic[i] = 2
        dic[i + 74] = 2

    dic = sorted([(key, value) for key, value in dic.items()])
    print(dict(dic))

# get_dic()
node_scale1 = [56, 130, 2, 116, 21, 68, 100, 58, 142, 19, 132, 27, 95, 139, 65, 93, 89, 101, 96, 99, 85, 7, 104, 69, 67,
               81, 128, 26, 44, 22, 147, 127, 11, 52, 25, 30, 103, 118, 54, 73, 15, 102, 141, 20, 29, 53, 80, 126, 94,
               28, 4, 90, 45, 6, 119, 16]
node_selected = [51, 116, 130, 57, 42, 27, 56, 76, 131, 93, 125, 2, 132, 58, 133, 95, 19, 59, 21, 142, 101, 147, 68, 100,
                 135, 129, 73, 67, 96, 88, 134, 139, 69, 85, 60, 22, 99, 143, 118, 44, 86, 65, 141, 94, 26, 38, 128, 137,
                 110, 11, 127, 81, 103, 46, 114, 104]
node_56inner20 = [130, 68, 56, 2, 21, 100, 142, 116, 58, 132, 19, 95, 93, 89, 139, 27, 101, 96, 65, 69]
node = [51, 116, 130, 57, 42, 27, 56, 76, 131, 93, 125, 2, 132, 58, 133, 95, 19, 59, 21, 142, 101, 147, 68, 100, 135, 129, 73, 67, 96, 88, 134, 139, 69, 85, 60, 22, 99, 143, 118, 44, 86, 65, 141, 94, 26, 38, 128, 137, 110, 11, 127, 81, 103, 46, 114, 104, 25, 15, 102, 7, 120, 29, 123, 20, 145, 89, 52, 115, 14, 112, 40, 126, 50, 108, 84, 30, 43, 119, 124, 90, 78, 54, 144, 77, 91, 62, 105, 28, 3, 31, 138, 72, 121, 106, 61, 49, 32, 111, 12, 34, 36, 136, 37, 97, 79, 146, 13, 70, 23, 6, 16, 92, 83, 10, 24, 53, 98, 80, 45, 66, 148, 63, 4, 109, 122, 117, 35, 113, 1, 82, 71, 8, 87, 18, 140, 48, 55, 75, 64, 5, 47, 17, 107, 39, 74, 9, 33, 41]
intersect = [2, 11, 19, 21, 22, 26, 27, 44, 56, 58, 65, 67, 68, 69, 73, 81, 85, 93, 94, 95, 96, 99, 100, 101, 103, 104, 116, 118, 127, 128, 130, 132, 139, 141, 142, 147]
def g_s(node):
    d = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in node:
        d[dic[i]] += 1
    print("gyrus: ", d[1])
    print("sulcus: ", d[2])
    print("G&S ", d[3])
    print("None: ", d[4])

# g_s(intersect)
# g_s(node_selected[:56])
# g_s(node_scale1)
# for i in range(0, 51, 10):
#     print(f"###Selected 56inner{i}--{i + 10}:")
#     g_s(node_selected[i:i+10])
#     # print(f"###56inner{i}--{i + 10}:")
#     # g_s(node_scale1[i:i+10])

import numpy as np
def plot_distribution(data, title):
    import matplotlib.pyplot as plt
    # 类别标签
    labels = ['Gyrus', 'Sulcus', 'Gyrus & Sulcus', 'Occipital pole\nTemporal pole']
    colors = ['#A0CEBD', '#9CA3DB', '#D8EC8A', '#DB9A7E']
    # 绘制堆叠柱形图
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(data[0]))
    segment_names = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-56']  # 段的名称

    for i, category in enumerate(data):
        plt.bar(np.arange(len(category)), category, bottom=bottom, label=labels[i], color=colors[i], edgecolor="black")
        bottom += category

    # 添加标签和标题
    plt.xlabel('Segments')
    plt.ylabel('Node Count')
    plt.title(title)
    plt.xticks(np.arange(len(data[0])), segment_names)
    plt.legend(loc='upper right')

    # 显示图形
    plt.grid(True)
    plt.show()

# 不同段的节点数量列表
segment_nodes_selected = [
    [2, 5, 1, 2],  # 第一段节点数量列表
    [3, 6, 1, 0],  # 第二段节点数量列表
    [4, 6, 0, 0],  # 第三段节点数量列表
    [3, 7, 0, 0],  # 第四段节点数量列表
    [6, 4, 0, 0],   # 第五段节点数量列表
    [2, 3, 1, 0]    # 第六段节点数量列表
]

segment_nodes_scale1 = [
    [3, 5, 1, 1],  # 第一段节点数量列表
    [7, 3, 0, 0],  # 第二段节点数量列表
    [4, 4, 2, 0],  # 第三段节点数量列表
    [4, 6, 0, 0],  # 第四段节点数量列表
    [6, 3, 1, 0],   # 第五段节点数量列表
    [2, 2, 2, 0]    # 第六段节点数量列表
]

# 转置数据以匹配每个颜色对应一个类别的节点数量
segment_nodes_transposed_selected = np.array(segment_nodes_selected).T.tolist()
segment_nodes_transposed_scale1 = np.array(segment_nodes_scale1).T.tolist()

# 绘制堆叠柱形图
# plot_distribution(segment_nodes_transposed_selected, 'Selected Node Distribution')
# plot_distribution(segment_nodes_transposed_scale1, 'Scale-1 Node Distribution')


def plot_compare():
    import numpy as np
    import matplotlib.pyplot as plt

    # 数据
    categories = ['KBRs', 'Hub', 'Common', 'Specific']
    gyrus = [20, 26, 16, 4]
    sulcus = [31, 23, 17, 14]
    gyrus_sulcus = [3, 6, 2, 1]
    occipital_temporal_pole = [2, 1, 1, 1]
    # 临时，少版
    categories = ['Common', 'Specific']
    gyrus = [16, 4]
    sulcus = [17, 14]
    # 颜色
    colors = ['#A0CEBD', '#9CA3DB', '#D8EC8A', '#DB9A7E']

    # 绘图
    plt.figure(figsize=(6, 6), dpi=600)

    bar_width = 0.5
    index = np.arange(len(categories))

    plt.bar(index, gyrus, bar_width, label='Gyrus', color=colors[0])
    plt.bar(index, sulcus, bar_width, bottom=gyrus, label='Sulcus', color=colors[1])
    # plt.bar(index, gyrus_sulcus, bar_width, bottom=np.array(gyrus) + np.array(sulcus), label='Gyrus & Sulcus',
    #         color=colors[2])
    # plt.bar(index, occipital_temporal_pole, bar_width,
    #         bottom=np.array(gyrus) + np.array(sulcus) + np.array(gyrus_sulcus), label='Occipital pole & Temporal pole',
    #         color=colors[3])

    plt.xlabel('Node Type')
    plt.ylabel('Count')
    plt.title('Comparison of Gyrus and Sulcus in Different Node Types')
    plt.xticks(index, categories)
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_compare()




# g_s([128, 4, 6, 15, 16, 25, 28, 29, 30, 45, 53, 54, 80, 81, 90, 94, 99, 102, 103, 119])