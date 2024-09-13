import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution, Linear


def normalize_A(A, symmetry=True):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)     #A+ A的转置
        d = torch.sum(A, 1)   #对A的第1维度求和
        d = 1 / torch.sqrt(d + 1e-10)    # d的-1/2次方
        D = torch.diag_embed(d)    # 对角阵
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L

# 返回一个chebyshev的adj列表（k=0是一个，1是一个，2是一个...）
def generate_cheby_adj(A, K, device):
     support = []
     for i in range(K):
         if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
            temp = torch.zeros(A.shape[1], A.shape[1], device = device)    # A.shape[1]是节点数
            # temp = torch.eye(A.shape[1])
            # temp = temp.to(device)
            support.append(temp)
         elif i == 1:
            support.append(A)
         else:
            temp = torch.matmul(2*A, support[i-1])
            support.append(temp-support[i-1])
     return support

# 用k阶切比雪夫多项式拟合(其实是K-1阶数、、模型里的K和切比雪夫多项式的K不是一个概念，阶数是排除了第0项的，而此处的K是单纯的有几项）
class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()  # https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):    # 01两个不同GCN层（01两个多项式相当于不同卷积核？）
            # xdim[2]:148
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x, L):
        device = x.device
        adj = generate_cheby_adj(L, self.K, device)   # adj =? chebyshev(L~)
        for i in range(len(self.gc1)):      #
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])    # 把两个拟合的结果相加即得用一阶切比雪夫拟合的GCN的结果
        result = F.relu(result)
        return result

# 【model_0】148个节点邻接矩阵做参数且随机初始化的
class DGCNN(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=7):  #
        # xdim: (batch_size*num_nodes*num_fea00-tures_in)
        # xdim = [batchsize, 148, 148]
        # k_adj: num_layers
        # k_adj: 切比雪夫多项式项数
        # num_out: num_features_out
        # num_out: 148
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  # 对第二维（第0维为batch_size)进行标准化
        self.fc1 = Linear(xdim[1] * num_out, 32)  # 相当于把每张图做了readout,其feature_size——all = node_num * feature_size_node -> 32
        # self.fc2=Linear(64, 32)
        self.fc3 = Linear(32, 8)
        self.fc4 = Linear(8, nclass)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]))  # 148*148
        nn.init.xavier_normal_(self.A)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # 因为第三维 才为特征维度
        L = normalize_A(self.A)  # A是自己设置的148*148的可训练参数  即：邻接矩阵 L = D~-1/2 A~ D~-1/2 归一化L得到拉普拉斯矩阵
        # L = self.A
        result = self.layer1(x, L)      # layer1 chebyshev自带了relu已经激活
        result = result.reshape(x.shape[0], -1)      # 第0维batchsize不变，后面的缩成一列（8*（148*148）） 8 * 21904
        result = F.elu(self.fc1(result))             # 8 * 32
        # result=F.relu(self.fc2(result))
        result = F.elu(self.fc3(result))             # 8 * 8
        result = self.fc4(result)                    # 8 * class #7   每一列代表的是class的概率
        return result, L


# 【model_1】148个节点邻接矩阵做参数且用结构信息初始化的
class DGCNN_1(nn.Module):
    def __init__(self, xdim, k_adj, num_out, s_adjmatrix, nclass=7):  #
        # xdim: (batch_size*num_nodes*num_features_in)
        # xdim = [batchsize, 148, 148]
        # k_adj: num_layers
        # k_adj: 切比雪夫多项式项数
        # num_out: num_features_out
        # num_out: 148
        super(DGCNN_1, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  # 对第二维（第0维为batch_size)进行标准化
        self.fc1 = Linear(xdim[1] * num_out, 32)
        # self.fc2=Linear(64, 32)
        self.fc3 = Linear(32, 8)
        self.fc4 = Linear(8, nclass)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())

        # 将 s_adjmatrix 转换为 PyTorch 张量
        s_adjmatrix_tensor = torch.from_numpy(s_adjmatrix)
        # 创建 nn.Parameter 对象并将其赋值为 s_adjmatrix_tensor
        self.A = nn.Parameter(s_adjmatrix_tensor)
        #nn.init.xavier_normal_(self.A) #使用 Xavier 正态分布进行初始化

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # 因为第三维 才为特征维度
        L = normalize_A(self.A)  # A是自己设置的148*148的可训练参数  即：邻接矩阵
        # L = self.A
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.elu(self.fc1(result))
        # result=F.relu(self.fc2(result))
        result = F.elu(self.fc3(result))
        result = self.fc4(result)
        return result, L

# 【model_2】148个节点邻接矩阵不做参数一直用用结构信息
class DGCNN_2(nn.Module):
    def __init__(self, xdim, k_adj, num_out, s_adjmatrix, nclass=7):
        # xdim: (batch_size*num_nodes*num_features_in)
        # xdim = [batchsize, 148, 148]
        # k_adj: num_layers
        # k_adj: 切比雪夫多项式项数
        # num_out: num_features_out
        # num_out: 148
        super(DGCNN_2, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  # 对第二维（第0维为batch_size)进行标准化
        self.fc1 = Linear(xdim[1] * num_out, 32)
        # self.fc2=Linear(64, 32)
        self.fc3 = Linear(32, 8)
        self.fc4 = Linear(8, nclass)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())

        # 将 s_adjmatrix 转换为 PyTorch 张量
        s_adjmatrix_tensor = torch.from_numpy(s_adjmatrix)
        # 赋值为 s_adjmatrix_tensor
        self.A = s_adjmatrix_tensor.to("cuda")

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # 因为第三维 才为特征维度
        L = normalize_A(self.A)  # A是自己设置的148*148的可训练参数  即：邻接矩阵
        # L = self.A
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.elu(self.fc1(result))
        # result=F.relu(self.fc2(result))
        result = F.elu(self.fc3(result))
        result = self.fc4(result)
        return result, L

# 【model_3】邻接矩阵不做参数一直用固定的全1，为了对比节点特征选择的情况
class DGCNN_3(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=7):
        # xdim: (batch_size*num_nodes*num_features_in)
        # xdim = [batchsize, 148, 148]
        # k_adj: num_layers
        # k_adj: 切比雪夫多项式项数
        # num_out: num_features_out
        # num_out: 148
        super(DGCNN_3, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  # 对第二维（第0维为batch_size)进行标准化
        self.fc1 = Linear(xdim[1] * num_out, 32)
        # self.fc2=Linear(64, 32)
        self.fc3 = Linear(32, 8)
        self.fc4 = Linear(8, nclass)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())

        # 将 全1矩阵 转换为 PyTorch 张量
        adjmatrix = np.ones((xdim[1], xdim[1]))
        adjmatrix_tensor = torch.from_numpy(adjmatrix)
        # 赋值为 adjmatrix_tensor
        self.A = adjmatrix_tensor.to("cuda")

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # 因为第三维 才为特征维度
        L = normalize_A(self.A)
        # L = self.A
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.elu(self.fc1(result))
        # result=F.relu(self.fc2(result))
        result = F.elu(self.fc3(result))
        result = self.fc4(result)
        return result, L
