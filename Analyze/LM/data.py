# LM
node2 = [90, 2, 131, 130, 57, 28, 67, 132, 20, 116, 89, 56, 100, 127, 42, 7, 81, 76, 53, 82, 129, 85, 86, 147, 11, 134, 15, 21, 26, 69, 4, 94, 51, 19, 16, 88, 93, 58, 102, 27, 141, 1, 22, 49, 34, 59, 45, 46, 103, 75, 73, 142, 136, 114, 120, 77, 126, 99, 44, 48, 52, 65, 8, 79, 18, 125, 17, 146, 29, 43, 123, 96, 128, 12, 68, 14, 139, 62, 115, 133, 40, 119, 91, 70, 108, 6, 87, 3, 60, 95, 31, 138, 35, 104, 124, 113, 54, 78, 38, 105, 145, 5, 50, 118, 13, 80, 55, 137, 117, 122, 143, 24, 92, 30, 71, 140, 101, 66, 112, 107, 33, 9, 63, 144, 109, 110, 36, 23, 111, 61, 37, 121, 39, 72, 148, 47, 25, 41, 83, 106, 74, 135, 98, 64, 97, 10, 32, 84]

scale1_56 = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]


print(node2[:56])   # 与56scale-1对比
# print(node2[:41])   # 与41scale-1对比（仅EL的scale-1节点）

# 读取all_top.mat，返回dataset_list
# dataset_list 0-7 分别对应 Resting  E G L M R S W
def get_data_list():
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
    return dataset_list

'''分析七类Scale-1节点'''

# 获取需要提取的节点index（并集）
def get_union_index(dataset_list):
    classes = [3, 4]  # LM
    # 取列表的并集
    dataset_index = []
    seen_elements = set()  # 创建一个空集合，用于存储已经出现的元素
    for i, list in enumerate(dataset_list):
        if i in classes:
            for index in list:
                if index not in seen_elements:
                    dataset_index.append(index)
                    seen_elements.add(index)

    union_index = sorted(dataset_index)  # 排序
    print(f"######EL类并集节点：" + f'{len(union_index)}')
    print(union_index)

    return union_index

def non_overlap(node_index, Selected):
    non_intersection_Scale_1 = []
    for a in node_index:
        for b in Selected:
            if a not in Selected:
                non_intersection_Scale_1.append(a)
    non_intersection_Scale_1 = list(set(non_intersection_Scale_1))
    print("###Scale-1中非交集： ", non_intersection_Scale_1)
    print(len(non_intersection_Scale_1))

    non_intersection_Selected = []
    for a in Selected:
        for b in node_index:
            if a not in node_index:
                non_intersection_Selected.append(a)
    non_intersection_Selected = list(set(non_intersection_Selected))
    print("###Selected中非交集： ", sorted(non_intersection_Selected))
    print(len(non_intersection_Selected))

if __name__ == "__main__":
    dataset_list = get_data_list()
    # print(dataset_list)
    scale1_41 = get_union_index(dataset_list)

    # LM
    print("###38:")
    print(len(scale1_41))
    print("交集：", set(scale1_41).intersection(set(node2[:38])))       # 有19个交集
    non_overlap(scale1_41, node2[:38])

    print("###56:")
    print("交集：", set(scale1_56).intersection(set(node2[:56])))       # 56有36个交集、、好家伙 多了15个节点能多那么多吗
    non_overlap(scale1_56, node2[:56])


