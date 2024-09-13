import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 创建数据框
# ELS
# data = {
#     "Category": ["E", "G", "L",  "M", "R", "S", "W", "RS"],
#     # "overlapping": [2, 4, 9, 6, 13, 8, 7, 1],
#     "Specific": [11, 1, 8, 4, 5, 10, 5, 0],
#     "Non-overlapping Hub": [0, 2, 0, 25, 2, 4, 2, 6]
# }
data = {
    "Category": ["E", "G", "L",  "M", "R", "S", "W", "RS"],
    # "overlapping": [2, 4, 9, 6, 13, 8, 7, 1],
    "Specific": [11, 1, 8, 4, 5, 10, 5, 0],
    "Non-overlapping Hub": [0, 2, 0, 25, 2, 4, 2, 6]
}
# # 7
# "Non-overlapping Select-56": [4, 3, 12, 4, 11, 13, 6, 1],
# "Non-overlapping Scale-1": [0, 2, 1, 19, 4, 5, 2, 11]
## EL
# "Non-overlapping Select-56": [9, 1, 25, 5, 8, 8, 4, 0],
# "Non-overlapping Scale-1": [1, 3, 2, 7, 6, 12, 2, 12]


df = pd.DataFrame(data)

# 转置数据框以便热力图显示
df = df.set_index("Category")

# 生成热力图
plt.figure(figsize=(8, 6), dpi=600)
sns.heatmap(df, annot=True, cmap="RdPu")
# plt.title("Comparison of Non-overlap Categories")
# plt.xlabel("Category")
# plt.ylabel("Region")
# plt.show()
plt.savefig("el")