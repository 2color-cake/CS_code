


classes = 7

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

# 1.获取需要提取的节点index（并集）（num控制是几分类）
def get_union_index(dataset_list, num):
    classes = []
    if num == 2:   # ES
        classes = [1, 6]
    elif num == 3: # ELS
        classes = [1, 3, 6]
    elif num == 4:
        classes = [1, 3, 6, 7]
    elif num == 5:
        classes = [1, 3, 4, 6, 7]
    elif num == 6:
        classes = [1, 3, 4, 5, 6, 7]
    elif num == 7:
        classes = [1, 2, 3, 4, 5, 6, 7]
    else:
        classes = [i for i in range(0, 8)]
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
    print(f"######{num}类并集节点：" + f'{len(union_index)}')
    print(union_index)

    return union_index

# 2.获取需要提取的节点index（交集）
# 获取需要提取的节点index（交集）
def get_intersection_index(dataset_list,num):
    classes = []
    if num == 2:  # ES
        classes = [1, 6]
    elif num == 3:  # ELS
        classes = [1, 3, 6]
    elif num == 4:
        classes = [1, 3, 6, 7]
    elif num == 5:
        classes = [1, 3, 4, 6, 7]
    elif num == 6:
        classes = [1, 3, 4, 5, 6, 7]
    elif num == 7:
        classes = [1, 2, 3, 4, 5, 6, 7]
    else:
        classes = [i for i in range(0, 8)]

    # 取列表的交集
    lists = []

    for i, list in enumerate(dataset_list):
        if i in classes:
            lists.append(list)
    # print(lists)
    intersection_nodes = set(lists[0])
    for list in lists:
        # print(list)
        intersection_nodes = intersection_nodes.intersection(set(list))

    print(f"######{num}类交集节点：" + f'{len(intersection_nodes)}')
    print(sorted(intersection_nodes))

    return intersection_nodes

# 3.获取差集
def get_div_index(dataset_list, num):
    union = get_union_index(dataset_list, num)
    intersection = get_intersection_index(dataset_list, num)
    print(f"######{num}类差集节点：{len(union) - len(intersection)}")
    print(set(union) - set(intersection))

if __name__ == "__main__":
    dataset_list = get_data_list()
    # get_union_index(dataset_list, classes)
    # print()
    # get_intersection_index(dataset_list, classes)
    # print()
    get_div_index(dataset_list, classes)
