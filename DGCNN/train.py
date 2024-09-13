import os

import torch
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import DGCNN, DGCNN_1, DGCNN_2, DGCNN_3

os.environ['TORCH_HOME'] = './' #setting the environment variable

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter  # 导入命令行参数解析工具
from utils import *    # 自己的导入数据用的工具包


# 检测是否有GPU可用，如果有就使用GPU，否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个命令行参数解析器
parser = ArgumentParser("UGformer", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

# 添加命令行参数，这些参数用于配置脚本的运行
parser.add_argument("--run_folder", default="../", help="运行文件夹")
parser.add_argument("--dataset", default=[
                                          "D:\DataSet\DS_task_rest\\task\DS_compare_fc_top20\\node\\EMOTION\\Pears_148_obj_",
                                          "D:\DataSet\DS_task_rest\\task\DS_compare_fc_top20\\node\\GAMBLING\\Pears_148_obj_",
                                          "D:\DataSet\DS_task_rest\\task\DS_compare_fc_top20\\node\\LANGUAGE\\Pears_148_obj_",
                                          "D:\DataSet\DS_task_rest\\task\DS_compare_fc_top20\\node\\MOTOR\\Pears_148_obj_",
                                          "D:\DataSet\DS_task_rest\\task\DS_compare_fc_top20\\node\\RELATIONAL\\Pears_148_obj_",
                                          "D:\DataSet\DS_task_rest\\task\DS_compare_fc_top20\\node\\SOCIAL\\Pears_148_obj_",
                                          "D:\DataSet\DS_task_rest\\task\DS_compare_fc_top20\\node\\WM\\Pears_148_obj_"
                                          ], help="数据集路径")  # dataset就是一个路径的列表，node和edge都要，最后那个“_”后面是节点的标号

parser.add_argument("--num_list", default="D:\DataSet\DS_task_rest\\task\DS_148\\list.txt", help="节点标号列表路径")  # num_list是所用的节点的标号，文件名中都有
parser.add_argument("--num_node", default=20, help="节点数")
parser.add_argument("--batch_size", default=8, type=int, help="批处理大小")
parser.add_argument('--fold_idx', type=int, default=7, help='折叠索引，取值1-9')
parser.add_argument('--epoch_num', type=int, default=100, help='epoch轮次数量')
parser.add_argument('--lr', default=0.001, help='学习率')
parser.add_argument('--k_adj', default=2, help='切比雪夫多项式项数')    # k小一点泛化强一点、目前来看
parser.add_argument('--ex_type', default=0, help='0为节点邻接矩阵做参数且随机初始化的；'
                                                 '1为节点邻接矩阵做参数且用结构信息初始化的；'
                                                 '2为节点邻接矩阵不做参数只做单一运算设置为结构信息邻接矩阵的'
                                                 '3为节点邻接矩阵不做参数只做单一运算统一设置为node*node的全1矩阵的'
                                                 '4为节点邻接矩阵做参数且用功能信息初始化的'
                                                 '5为矩阵融合'
                                                 '6为节点邻接矩阵不做参数且用功能信息')

args = parser.parse_args()  # 解析命令行参数并存储在args变量中
print(args)  # 打印命令行参数的值

s_folder_path = r"D:\DataSet\DS_task_rest\task\DS_56_Selected\edge"
f_folder_path = r"D:\DataSet\DS_task_rest\task\DS_56_Selected\node"

train_loss = []   # 画图
test_loss = []
train_accuracy = []
test_accuracy = []
test_labels = []
test_pred = []

'''
seed = 121314
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
'''
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 记录实验信息,返回类别名称用于后续给生成的图命名
def record():
    categories = ""
    for i, ds in enumerate(args.dataset):
            if i < (len(args.dataset)):
                categories += ds.split("\\")[6][0]
            if i != (len(args.dataset)-1):
                    categories += "_"
    return categories

# 绘制曲线图
def draw_figure(lst, categories_name, figure_name, ylabel):
    plt.figure()
    lst = [tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in lst]
    plt.plot(lst)
    plt.title(figure_name)
    plt.xlabel('Epoch')
    # 设置 x 轴刻度间隔为每隔5个 epoch 显示一个刻度
    interval = 5
    plt.xticks(np.arange(0, len(lst), interval), np.arange(0, len(lst), interval).astype(int))

    plt.ylabel(ylabel)
    plt.savefig(f"{categories_name}---{figure_name}.jpg")
#  绘制混淆矩阵
def draw_confusionMatrix(test_labels, test_pred):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, test_pred)

    # 定义类别名称
    classes = ['E', 'G', 'L', 'M', 'R', 'S', 'W']

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
# 计算精确率和召回率
def cal_Precision_Recall(test_labels, test_pred):
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(test_labels, test_pred, average='weighted') #average参数设置为'weighted'来计算加权平均值，其中每个类别的权重是它在测试集中的比例
    recall = recall_score(test_labels, test_pred, average='weighted')

    print('Precision:', precision)
    print('Recall:', recall)


# 加载数据
def load_fMRIdata():
    print("Loading data...")  # 输出提示信息，表示正在加载数据

    graphs = []  # S2VGraph类型的无向图集合，每个S2VGraph里包含一个g即networkX类型的图
    lable = []   # 对应上面无向图的类标
    # 循环遍历数据集路径，加载数据
    for i in range(len(args.dataset)):
        graphs_c = load_data(args.dataset, args.num_list, i)  # i是当前所用的dataset路径序号
        lable_c = (np.zeros(
            len(graphs_c)) + i).tolist()  # np.zeros生成一个graphs_c那么长的列表，即对应每一张图都有一个标签（具体的标签其实是i,因为广播特性矩阵加一个i即每个元素都加了i，最后再转化为列表）
        graphs = graphs + graphs_c
        lable = lable + lable_c

    th = 0
    # 根据阈值th对图数据进行处理
    # 比阈值小的相关性都设为0
    for i in range(len(graphs)):
        graphs[i].node_features[graphs[i].node_features < th] = 0

    train_idx, test_idx, train_graphs, test_graphs = separate_data(graphs, args.fold_idx)  # 分割训练集和测试集

    # 把所有train数据和test数据拼成一张大表并转为tensor_3d 三维张量，第一维度给成序号
    def concat(S2VGraph_a):
        tensor_3d = torch.empty((len(S2VGraph_a), args.num_node, args.num_node))
        for i in range(len(S2VGraph_a)):
            tensor_3d[i] = torch.from_numpy(S2VGraph_a[i].node_features)

        # tensor_3d现在是一个三维张量，第一维度是批次序号
        # print(tensor_3d.shape)
        # exit(0)
        return tensor_3d

    train_all_ds = concat(train_graphs)
    test_all_ds = concat(test_graphs)

    train_label = [lable[i] for i in train_idx]  # 训练集标签
    test_label = [lable[i] for i in test_idx]  # 测试集标签

    feature_dim_size = graphs[0].node_features.shape[1]  # 获取节点特征的维度大小  (node_features也就是FC矩阵，行号对应具体点号，列数对应的是特征维度）

    print("######feature_dim_size:"+str(feature_dim_size))
    print("Loading data... finished!")  # 输出加载数据完成的信息
    print()

    return feature_dim_size, train_all_ds, train_label, test_all_ds, test_label


def train(train_iter, test_iter, model, criterion, optimizer, epoch_num):
    # Train
    print('began training on', device, '...')

    acc_test_best = 0.0
    n = 0
    L = np.zeros((args.num_node, args.num_node))   # 存中间邻接矩阵的
    for ep in range(epoch_num):
        model.train()
        n += 1
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        accuracy = 0.

        for images, labels in train_iter:

            images = images.float().to(device)
            labels = labels.to(device)

            # Compute loss & accuracy
            output, L = model(images)    # 这里会多返回一个L邻接矩阵
            loss = criterion(output, labels.long())

            pred = output.argmax(dim=1)  # output张量每行最大值的索引，每行都是num_class个维度，哪一维度值大就分到哪一维度对应的类

            correct += (pred == labels).sum().item()
            total += len(labels)
            accuracy = correct / total

            total_loss += loss
            loss.backward()
            #scheduler.step()
            optimizer.step()
            #print(optimizer.state_dict)
            optimizer.zero_grad()
            print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
                                                                    batch_id,
                                                                    total_loss / batch_id,
                                                                    accuracy))
            batch_id += 1

        # 训练到最后存储中间生成的邻接矩阵
        if ep == (epoch_num - 1):
            save_adjmatrix_L(L, "adj_148_generate.jpg")
            print(L)  #####################

        train_loss.append(total_loss/args.batch_size)################
        train_accuracy.append(accuracy)##############
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

        acc_test, loss_test = evaluate(test_iter, criterion, model, ep)
        test_loss.append(loss_test)
        test_accuracy.append(acc_test)   ##############

        if acc_test >= acc_test_best:
            n = 0
            acc_test_best = acc_test
            model_best = model

        # 学习率逐渐下降，容易进入局部最优，当连续10个epoch没有跳出，且有所下降，强制跳出
        # if n >=  num_epochs//10 and acc_test < 0.99: 
        if n >=  epoch_num//10 and acc_test < acc_test_best-0.1:
            print('#########################reload#########################')
            n = 0
            model = model_best
        # find best test acc model in all epoch(not last epoch)

        print('>>> best test Accuracy: {}'.format(acc_test_best))

    return acc_test_best, L

def evaluate(test_iter, criterion, model, ep):
    global test_labels, test_pred
    # Eval
    print('began test on', device, '...')
    model.eval()
    correct, total = 0, 0
    loss_test = 0
    a = []
    b = []
    for images, labels in test_iter:
        # Add channels = 1
        images = images.float().to(device)
    
        # Categogrical encoding
        labels = labels.to(device)
        
        output, _ = model(images)
        loss = criterion(output, labels.to(device, dtype=torch.long))  # 计算损失
        loss_test += loss.item()  # 累加测试损失

        pred = output.argmax(dim=1)
        if ep == args.epoch_num-1:
            test_labels.append([int(x) for x in labels.tolist()])
            test_pred.append([int(x) for x in pred.tolist()])
        correct += (pred == labels).sum().item()
        total += len(labels)

        #a.append([int(x) for x in labels.tolist()])
        #b.append([int(x) for x in pred.tolist()])
    print('test Accuracy: {}'.format(correct / total))
    '''
    if (correct/total) >= 0.84:
        a = [i for sublist in a for i in sublist]
        b = [i for sublist in b for i in sublist]
        draw_confusionMatrix(a, b)
        cal_Precision_Recall(a, b)
        exit(0)
    '''
    return correct / total, loss_test/len(test_iter)

# 返回一个数据加载器，此处分别返回了train_iter和test_iter
def load_dataloader(data_train, data_test, label_train, label_test, batch_size = args.batch_size):  # 默认为args里的batchsize
    '''
    DataLoader是一个数据加载器，用于从Dataset对象中加载数据并生成小批量的数据。它提供了数据的批量加载、并行加载和数据重排的功能。
    返回一个数据加载器，此处分别返回了train和test的iter
    '''
    train_iter = DataLoader(dataset=fMRIDataSet(data_train, label_train),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)

    test_iter = DataLoader(dataset=fMRIDataSet(data_test, label_test),
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=1)

    return train_iter, test_iter

# 把先验的结构矩阵导入（平均了98个subject的结果）
def load_s_adjmatrix():
    node_num = args.num_node
    graph_num = 98

    # 文件夹所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(s_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # 存图邻接矩阵的字典
    graphs = {}
    for i in range(graph_num):
        graphs[i] = np.zeros((node_num, node_num), dtype=int)  # 创建邻接矩阵数组

    # 全部存入
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:  # 读取文件
            lines = file.readlines()
        # 解析文件内容
        edge_num = int(lines[0])  # 边的数量
        edges = [list(map(int, line.split())) for line in lines[1:]]  # [[1, 5], [1, 15], [1, 16],.....]的列表
        # 填充邻接矩阵
        for edge in edges:
            node_a, node_b = edge
            graphs[i][node_a - 1, node_b - 1] = 1  # 减1是因为节点序号从1开始
            graphs[i][node_b - 1, node_a - 1] = 1  # 对称

    avg_adjMatrix = np.zeros((node_num, node_num))  # 总的
    # 做平均再可视化
    for i in range(graph_num):
        avg_adjMatrix += graphs[i]
    avg_adjMatrix /= graph_num
    return avg_adjMatrix

# 把先验的直接平均的功能矩阵导入（平均了7类的）# 现在的folder_path是没换的148的
def load_f_adjmatrix():
    node_num = args.num_node
    # 文件夹所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(f_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # print(file_paths)

    matrix = np.zeros((node_num, node_num))
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = [list(map(float, line.strip().split())) for line in lines[1:]]
        matrix += np.array(data)

    # print(len(file_paths))
    adj_function = matrix / len(file_paths)
    print(adj_function)
    return adj_function


if __name__ == '__main__':


    # 0: 27
    # 1: 29
    # 2: 26
    # 3: 30
    # 4: 35
    # 5: 32
    # 6: 31
    # print(test_label)
    # print("0: ", test_label.count(0.0))
    # print("1: ", test_label.count(1.0))
    # print("2: ", test_label.count(2.0))
    # print("3: ", test_label.count(3.0))
    # print("4: ", test_label.count(4.0))
    # print("5: ", test_label.count(5.0))
    # print("6: ", test_label.count(6.0))

    # '''
    # 测试数据一直没变（shuffle那边有seed）
    feature_dim_size, train_data, train_label, test_data, test_label = load_fMRIdata()


    xdim = [args.batch_size, args.num_node, feature_dim_size]
    k_adj = args.k_adj  ### 切比雪夫多项式项数
    num_out = args.num_node

    if args.ex_type == 0:
        model = DGCNN(xdim, k_adj, num_out).to(device)
    elif args.ex_type == 1:
        s_adjmatrix = load_s_adjmatrix()
        model = DGCNN_1(xdim, k_adj, num_out, s_adjmatrix).to(device)
    elif args.ex_type == 2:
        s_adjmatrix = load_s_adjmatrix()
        model = DGCNN_2(xdim, k_adj, num_out, s_adjmatrix).to(device)
    elif args.ex_type == 3:
        model = DGCNN_3(xdim, k_adj, num_out).to(device)
    elif args.ex_type == 4:
        f_adjmatrix = load_f_adjmatrix()
        model = DGCNN_1(xdim, k_adj, num_out, f_adjmatrix).to(device)
    elif args.ex_type == 5:     # 融合初始化
        # s_adjmatrix = load_s_adjmatrix()
        f_adjmatrix = load_f_adjmatrix()
        matrix = f_adjmatrix * f_adjmatrix
        model = DGCNN_1(xdim, k_adj, num_out, matrix).to(device)
    elif args.ex_type == 6:   # 功能初始化且不做参数，一直训练
        s_adjmatrix = load_f_adjmatrix()
        model = DGCNN_2(xdim, k_adj, num_out, s_adjmatrix).to(device)

    criterion = nn.CrossEntropyLoss().to(device)  # 使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=0.0001)

    epoch_num = args.epoch_num

    train_iter, test_iter = load_dataloader(train_data, test_data, train_label, test_label)
    acc_test_best, L = train(train_iter, test_iter, model, criterion, optimizer, epoch_num)

    print("acc_test_best:" + str(acc_test_best))


    categories = record()
    log_filename = 'record2.log'
    # 打开文件以进行追加写入
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\n###### Categories: {categories}\n")
        log_file.write(
            f"   【EX_type】:{args.ex_type}  【Node_num】：{args.num_node}"
            f"\n   epoch_num: {args.epoch_num}   Batch_size: {args.batch_size}  Lr: {args.lr}  K_adj: {args.k_adj}  train/test: {args.fold_idx}:{10 - args.fold_idx}\n")
        train_loss_N_tensor = [tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in train_loss]
        train_loss_lst =  [item.item() for item in train_loss_N_tensor]
        log_file.write(f"   train_loss:{train_loss_lst}\n")
        log_file.write(f"   train_accuracy:{train_accuracy}\n")
        log_file.write(f"   test_accuracy:{test_accuracy}\n")
        log_file.write(f"   test_loss:{test_loss}\n")
        log_file.write(f">>>>>>>Best Test Accuracy:{acc_test_best}\n")
        log_file.write(f'>>>>>>>Mean Test Accuracy:{sum(test_accuracy)/len(test_accuracy)}\n')
        log_file.write(f">>>>>>>Mean last_15  Test Accuracy:{sum(test_accuracy[-15:])/len(test_accuracy[-15:])}\n")

    L_filename = 'L.txt'

    # 设置 NumPy 数组打印选项，以显示所有元素
    np.set_printoptions(threshold=np.inf)

    # 将张量 L 分离，并转换为 NumPy 数组
    L_np = L.detach().cpu().numpy()

    # 将每个 [] 内容占据一行的形式生成字符串，并写入文件
    with open(L_filename, 'w') as file:
        for row in L_np:
            row_str = ' '.join([str(value) for value in row])
            file.write(f"{row_str}\n")

    ### 评估图绘制
    draw_figure(train_loss, categories, "Train_loss", "Loss")
    draw_figure(train_accuracy, categories, "Train_accuracy", "Accuracy")
    draw_figure(test_accuracy, categories, "Test_accuracy", "Accuracy")
    draw_figure(test_loss, categories, "Test_loss", "Loss")
    # 混淆矩阵
    test_labels = [i for sublist in test_labels for i in sublist]
    test_pred = [i for sublist in test_pred for i in sublist]
    draw_confusionMatrix(test_labels, test_pred)
    cal_Precision_Recall(test_labels, test_pred)


    '''
    # 需要30遍
    for i in range(1, 31):
        feature_dim_size, train_data   , train_label, test_data, test_label = load_fMRIdata()

        xdim = [args.batch_size, args.num_node, feature_dim_size]
        k_adj = args.k_adj  ### 切比雪夫多项式项数
        num_out = args.num_node

        if args.ex_type == 0:
            model = DGCNN(xdim, k_adj, num_out).to(device)
        elif args.ex_type == 1:
            s_adjmatrix = load_s_adjmatrix()
            model = DGCNN_1(xdim, k_adj, num_out, s_adjmatrix).to(device)
        elif args.ex_type == 2:
            s_adjmatrix = load_s_adjmatrix()
            model = DGCNN_2(xdim, k_adj, num_out, s_adjmatrix).to(device)
        elif args.ex_type == 3:
            model = DGCNN_3(xdim, k_adj, num_out).to(device)
        elif args.ex_type == 4:
            f_adjmatrix = load_f_adjmatrix()
            model = DGCNN_1(xdim, k_adj, num_out, f_adjmatrix).to(device)
        elif args.ex_type == 5:  # 融合初始化
            s_adjmatrix = load_s_adjmatrix()
            f_adjmatrix = load_f_adjmatrix()
            matrix = s_adjmatrix * f_adjmatrix
            model = DGCNN_1(xdim, k_adj, num_out, matrix).to(device)

        criterion = nn.CrossEntropyLoss().to(device)  # 使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=0.0001)

        epoch_num = args.epoch_num

        train_iter, test_iter = load_dataloader(train_data, test_data, train_label, test_label)
        acc_test_best, L = train(train_iter, test_iter, model, criterion, optimizer, epoch_num)

        print("acc_test_best:" + str(acc_test_best))

        categories = record()
        log_filename = 'record2.log'
        # 打开文件以进行追加写入
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n###### Categories: {categories}\n")
            log_file.write(
                f"   【EX_type】:{args.ex_type}  【Node_num】：{args.num_node}"
                f"\n   epoch_num: {args.epoch_num}   Batch_size: {args.batch_size}  Lr: {args.lr}  K_adj: {args.k_adj}  train/test: {args.fold_idx}:{10 - args.fold_idx}\n")
            train_loss_N_tensor = [tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for
                                   tensor in train_loss]
            train_loss_lst = [item.item() for item in train_loss_N_tensor]
            log_file.write(f"   train_loss:{train_loss_lst}\n")
            log_file.write(f"   train_accuracy:{train_accuracy}\n")
            log_file.write(f"   test_accuracy:{test_accuracy}\n")
            log_file.write(f">>>>>>>Best Test Accuracy:{acc_test_best}\n")
            log_file.write(f'>>>>>>>Mean Test Accuracy:{sum(test_accuracy) / len(test_accuracy)}\n')
            log_file.write(
                f">>>>>>>Mean last_15  Test Accuracy:{sum(test_accuracy[-15:]) / len(test_accuracy[-15:])}\n")

        L_filename = f'symmetery\\LM-L148\\L{i}.txt'



        # 设置 NumPy 数组打印选项，以显示所有元素
        np.set_printoptions(threshold=np.inf)

        # 将张量 L 分离，并转换为 NumPy 数组
        L_np = L.detach().cpu().numpy()

        # 将每个 [] 内容占据一行的形式生成字符串，并写入文件.

        with open(L_filename, 'w') as file:
            for row in L_np:
                row_str = ' '.join([str(value) for value in row])
                file.write(f"{row_str}\n")

        draw_figure(train_loss, categories, "Train_loss", "Loss")
        draw_figure(train_accuracy, categories, "Train_accuracy", "Accuracy")
        draw_figure(test_accuracy, categories, "Test_accuracy", "Accuracy")
    '''
