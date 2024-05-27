import itertools
import random
import pandas as pd

# 初始查询
queries = [
    "计算第二张表的A列总和",
    "计算索引为2的表的A列总和",
    "计算SHEET2的A列之和",
    "计算当前表的A列总和"
]

# 同义词替换和句式变换
synonym_queries = [
    "求第二张表的A列总和",
    "计算第二个表的A列总和",
    "求索引为2的表的A列总和",
    "计算索引2的表的A列总和",
    "求SHEET2的A列总和",
    "计算SHEET2表的A列总和",
    "求当前表的A列总和",
    "计算此表的A列总和"
]

# 生成不同数值和表名的组合
numbers = ["第一", "第二", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十"]
indices = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
sheet_names = ["SHEET1", "SHEET2", "SHEET3", "SHEET4", "SHEET5", "SHEET6", "SHEET7", "SHEET8", "SHEET9", "SHEET10"]
columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

new_queries = []
for num, idx, sheet, col in itertools.product(numbers, indices, sheet_names, columns):
    new_queries.append(f"计算{num}张表的{col}列总和")
    new_queries.append(f"计算索引为{idx}的表的{col}列总和")
    new_queries.append(f"计算{sheet}的{col}列总和")
    new_queries.append(f"计算当前表的{col}列总和")

# 将所有查询组合在一起
all_queries = queries + synonym_queries + new_queries

# 标注查询
total_samples = len(all_queries)
num_need_lookup = int(total_samples * 0.8)  # 需要进一步执行查找操作的数量
num_no_lookup = total_samples - num_need_lookup  # 不需要进一步执行查找操作的数量

# 随机选择需要进一步执行查找操作的查询
random.seed(42)  # 固定随机种子以便结果可复现
need_lookup_indices = random.sample(range(total_samples), num_need_lookup)

# 标注数据
labeled_data = []
for i, query in enumerate(all_queries):
    if i in need_lookup_indices:
        labeled_data.append((query, 1))  # 1表示需要进一步执行查找操作
    else:
        labeled_data.append((query, 0))  # 0表示不需要进一步执行查找操作

# 创建DataFrame
df = pd.DataFrame(labeled_data, columns=['Query', 'Label'])

# 保存为CSV文件
csv_file_path = './labeled_queries_expanded.csv'
df.to_csv(csv_file_path, index=False)

csv_file_path
