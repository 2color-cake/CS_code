# EL
node2 = [130, 19, 58, 132, 21, 57, 56, 131, 93, 20, 2, 51, 136, 59, 49, 95, 129, 125, 60, 76, 94, 27, 11, 142, 89, 22, 134, 4, 133, 67, 85, 50, 17, 96, 65, 68, 61, 127, 116, 86, 15, 7, 99, 28, 100, 75, 43, 34, 42, 18, 146, 45, 62, 126, 3, 53, 111, 135, 48, 35, 69, 139, 83, 8, 147, 88, 137, 52, 81, 143, 5, 108, 77, 87, 14, 73, 123, 26, 66, 90, 13, 64, 104, 102, 118, 6, 120, 138, 101, 141, 82, 44, 1, 103, 140, 9, 24, 124, 105, 33, 40, 72, 79, 74, 31, 30, 121, 12, 23, 92, 119, 117, 98, 10, 29, 113, 63, 80, 71, 39, 148, 144, 109, 107, 106, 114, 78, 32, 70, 145, 38, 25, 112, 55, 91, 46, 16, 54, 36, 97, 47, 41, 84, 37, 122, 128, 110, 115]
# ES
node2 = [133, 59, 101, 27, 69, 100, 131, 147, 19, 67, 57, 93, 130, 132, 142, 56, 143, 141, 88, 104, 58, 2, 125, 60, 134, 128, 112, 68, 76, 116, 26, 110, 137, 114, 111, 129, 51, 120, 87, 73, 94, 95, 90, 32, 102, 42, 124, 85, 11, 46, 103, 49, 144, 99, 105, 86, 21, 127, 30, 106, 121, 7, 146, 113, 63, 89, 28, 50, 135, 145, 20, 29, 126, 123, 61, 71, 5, 22, 139, 10, 31, 37, 40, 119, 140, 138, 36, 54, 14, 91, 65, 108, 84, 78, 79, 118, 82, 45, 107, 122, 39, 80, 44, 72, 62, 23, 83, 52, 38, 3, 48, 4, 96, 12, 15, 98, 115, 66, 13, 43, 8, 74, 53, 24, 81, 109, 9, 70, 92, 77, 75, 97, 25, 16, 34, 1, 136, 148, 47, 6, 41, 35, 17, 64, 55, 33, 18, 117]

scale1_56 = [2, 4, 6, 7, 11, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 44, 45, 52, 53, 54, 56, 58, 65, 67, 68, 69, 73, 80, 81, 85, 89, 90, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 116, 118, 119, 126, 127, 128, 130, 132, 139, 141, 142, 147]


print(node2[:56])   # 与56scale-1对比
print(node2[:41])   # 与41scale-1对比（仅EL的scale-1节点）

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
    classes = [1, 3]  # EL
    classes = [1, 6]  # ES
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

    # ES倒是有19个交集、、
    print("###37:")
    print(len(scale1_41))
    print("交集：", set(scale1_41).intersection(set(node2[:37])))       # 只有14个交集、、好少
    non_overlap(scale1_41, node2[:37])

    print("###56:")
    print("交集：", set(scale1_56).intersection(set(node2[:56])))       # 56有32个交集、、好家伙 多了15个节点能多那么多吗
    non_overlap(scale1_56, node2[:56])


