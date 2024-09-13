
node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]
node = [51, 116, 130, 57, 42, 27, 56, 76, 131, 93, 125, 2, 132, 58, 133, 95, 19, 59, 21, 142, 101, 147, 68, 100, 135, 129, 73, 67, 96, 88, 134, 139, 69, 85, 60, 22, 99, 143, 118, 44, 86, 65, 141, 94, 26, 38, 128, 137, 110, 11, 127, 81, 103, 46, 114, 104, 25, 15, 102, 7, 120, 29, 123, 20, 145, 89, 52, 115, 14, 112, 40, 126, 50, 108, 84, 30, 43, 119, 124, 90, 78, 54, 144, 77, 91, 62, 105, 28, 3, 31, 138, 72, 121, 106, 61, 49, 32, 111, 12, 34, 36, 136, 37, 97, 79, 146, 13, 70, 23, 6, 16, 92, 83, 10, 24, 53, 98, 80, 45, 66, 148, 63, 4, 109, 122, 117, 35, 113, 1, 82, 71, 8, 87, 18, 140, 48, 55, 75, 64, 5, 47, 17, 107, 39, 74, 9, 33, 41]
Selected = node[0:56]

for i in range(1, 6):
    common = len(set(Selected[(i-1)*10:i*10]).intersection(set(node_index)))

    print(f"#####top{10*i}个区域权重列表里：")
    print(f"Common：{common}个\n Specific：{10*i - common - (i-1)*10}个")



def draw():
    import matplotlib.pyplot as plt

    node_index = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67,
                  68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127,
                  128, 130, 132, 139, 141, 142, 147]
    node = [51, 116, 130, 57, 42, 27, 56, 76, 131, 93, 125, 2, 132, 58, 133, 95, 19, 59, 21, 142, 101, 147, 68, 100,
            135, 129, 73, 67, 96, 88, 134, 139, 69, 85, 60, 22, 99, 143, 118, 44, 86, 65, 141, 94, 26, 38, 128, 137,
            110, 11, 127, 81, 103, 46, 114, 104, 25, 15, 102, 7, 120, 29, 123, 20, 145, 89, 52, 115, 14, 112, 40, 126,
            50, 108, 84, 30, 43, 119, 124, 90, 78, 54, 144, 77, 91, 62, 105, 28, 3, 31, 138, 72, 121, 106, 61, 49, 32,
            111, 12, 34, 36, 136, 37, 97, 79, 146, 13, 70, 23, 6, 16, 92, 83, 10, 24, 53, 98, 80, 45, 66, 148, 63, 4,
            109, 122, 117, 35, 113, 1, 82, 71, 8, 87, 18, 140, 48, 55, 75, 64, 5, 47, 17, 107, 39, 74, 9, 33, 41]
    Selected = node[0:56]

    # 创建top ranges的字符串表示，并添加51-56
    top_ranges_str = [f'{10 * i - 9}-{10 * i}' for i in range(1, 6)] + ['51-56']
    common_counts = []
    specific_counts = []

    # 计算前五个范围的Common和Specific
    for i in range(1, 6):
        top_elements = Selected[(i - 1) * 10:i * 10]
        common = len(set(top_elements).intersection(set(node_index)))
        specific = 10 * i - common - (i - 1) * 10
        common_counts.append(common)
        specific_counts.append(specific)

    # 计算51-56范围的Common和Specific
    top_elements_51_56 = Selected[50:56]
    common_51_56 = len(set(top_elements_51_56).intersection(set(node_index)))
    specific_51_56 = len(top_elements_51_56) - common_51_56
    common_counts.append(common_51_56)
    specific_counts.append(specific_51_56)

    # 可视化
    fig, ax = plt.subplots(dpi=600)

    index = range(len(top_ranges_str))
    bar_width = 0.7  # 设置柱状图的宽度

    # 定义颜色
    common_color = '#CED9F8'
    specific_color = '#F4D7AF'

    # 绘制堆叠柱状图
    p1 = plt.bar(index, common_counts, bar_width, color=common_color, label='Common', edgecolor='grey')
    p2 = plt.bar(index, specific_counts, bar_width, bottom=common_counts, color=specific_color, label='Specific', edgecolor='grey')

    # plt.xlabel('Top N Elements')
    plt.ylabel('Counts')
    plt.title('Common vs Specific Counts')
    plt.xticks(index, top_ranges_str)
    plt.legend()

    plt.tight_layout()
    plt.show()


draw()