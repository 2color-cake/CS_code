import pandas as pd
import matplotlib.pyplot as plt

def draw_accuracy_model(data):
    df = pd.DataFrame(data)
    # 设置颜色
    colors = ['#6495ED', '#FFC125',  '#7CCD7C']
    # 绘制条形图
    df.plot(x='Model', kind='bar', edgecolor="blue", figsize=(13, 8), rot=45, color = colors)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy in Different Experiments')
    plt.legend(title='Experiment')

    # 显示图表
    plt.show()
    plt.savefig("img-accuracy\\7.jpg")


# 假设data是你整理好的数据集，包含模型、实验条件和准确率
data7 = {
        'Model': ['DGCNN', 'ResNet-50', 'ResNet-18', 'LeNet-5', 'SVM', 'KNN', 'LR', 'DecisionTree'],
        'All(148)':   [0.8415873015873017, 0.6700617283950616, 0.708641975308642,  0.7762345679012346, 0.8411721146725906, 0.5028985507246377, 0.8586268909340952, 0.4926901512747276],
        'KBRs(56)': [0.8777777777777781, 0.7388888888888889, 0.7657407407407406, 0.7478395061728396, 0.8585951549772559, 0.6136464614408125, 0.8644557283402093, 0.4634930709827569],
        'Random(56)': [0.6172857142857144, 0.4885802469135802, 0.5237654320987654, 0.550925925925926,  0.6676927959377975, 0.3455093621072675, 0.6953982862583307, 0.3308896646567227],
    }

data2 = {
    'Model': ['DGCNN', 'ResNet-50', 'ResNet-18', 'LeNet-5', 'SVM', 'KNN (k=8)', 'LR', 'DecisionTree'],
    'All(148)': [0.9513, 0.9363, 0.9241, 0.9460, 0.9645, 0.8674, 0.9645, 0.8419],
    'KBRs(56)': [0.9524, 0.9007, 0.9077, 0.9213, 0.9695, 0.9337, 0.9695, 0.8168],
    'Random(56)': [0.9185, 0.8595, 0.7975, 0.8237, 0.8831, 0.7042, 0.8882, 0.7199]
}
data3 = {
    'Model': ['DGCNN', 'ResNet-50', 'ResNet-18', 'LeNet-5', 'SVM', 'KNN (k=8)', 'LR', 'DecisionTree'],
    'All(148)': [0.9553, 0.8931, 0.8428, 0.8796, 0.9456, 0.7961, 0.9660, 0.7721],
    'KBRs(56)': [0.9582, 0.8914, 0.8852, 0.8870, 0.9558, 0.8844, 0.9592, 0.7006],
    'Random(56)': [0.8454, 0.7653, 0.8162, 0.7940, 0.8946, 0.6363, 0.8944, 0.6532]
}
data4 = {
    'Model': ['DGCNN', 'ResNet-50', 'ResNet-18', 'LeNet-5', 'SVM', 'KNN (k=8)', 'LR', 'DecisionTree'],
    'All(148)': [0.9115, 0.7569, 0.8655, 0.8516, 0.9387, 0.7218, 0.9438, 0.6658],
    'KBRs(56)': [0.9003, 0.8437, 0.8124, 0.8197, 0.9337, 0.8469, 0.9464, 0.6965],
    'Random(56)': [0.8336, 0.7336, 0.7826, 0.7861, 0.8520, 0.5561, 0.8698, 0.5967]
}
data5 = {
    'Model': ['DGCNN', 'ResNet-50', 'ResNet-18', 'LeNet-5', 'SVM', 'KNN (k=8)', 'LR', 'DecisionTree'],
    'All(148)': [0.8862, 0.6601, 0.8402, 0.8276, 0.9, 0.6755, 0.9102, 0.6122],
    'KBRs(56)': [0.9125, 0.735380, 0.798684, 0.7613, 0.9163, 0.7653, 0.9122, 0.5776],
    'Random(56)': [0.7596, 0.6256, 0.6760, 0.6424, 0.8102, 0.5286, 0.8367, 0.5102]
}
data6 = {
    'Model': ['DGCNN', 'ResNet-50', 'ResNet-18', 'LeNet-5', 'SVM', 'KNN (k=8)', 'LR', 'DecisionTree'],
    'All(148)': [0.8744, 0.725585, 0.813596, 0.835526, 0.8725, 0.5834, 0.8963, 0.5323],
    'KBRs(56)': [0.8432, 0.814474, 0.780994, 0.795175, 0.8708, 0.7195, 0.8844, 0.5426],
    'Random(56)': [0.7311, 0.573309, 0.556160, 0.597464, 0.7279, 0.4354, 0.7790, 0.4182]
}
draw_accuracy_model(data2)
