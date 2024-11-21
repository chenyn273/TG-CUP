import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
from statsmodels.stats.multitest import multipletests

# 读取数据函数


def read_data(file_path):
    # 读取文件，假设数据是单行且用逗号分隔
    data = pd.read_csv(file_path, header=None, sep=',')
    return data.iloc[0].values  # 提取第一行的数据，并转为 numpy 数组


# 从四个文件读取数据
data1 = read_data('/Users/chenyn/Downloads/METEOR-2.txt')
data2 = read_data('prediction/cup/METEOR')
data3 = read_data('/Users/chenyn/Downloads/METEOR.txt')
data4 = read_data('prediction/hat/METEOR')

# 执行多个 Wilcoxon 检验并收集 p 值
p_values = []
for (d1, d2) in [(data1, data2), (data1, data3), (data1, data4)]:
    statistic, p_value = wilcoxon(d1, d2)
    p_values.append(p_value)

# 打印原始 p 值
print("原始 p 值:", p_values)

# 进行 p 值校正（使用 Benjamini-Hochberg 方法）
rejected, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

# 打印校正后的 p 值
print("校正后的 p 值:", p_values_corrected)

# 判断哪些检验被拒绝（显著）
for i, (p_value, corrected_p) in enumerate(zip(p_values, p_values_corrected)):
    print(f"检验 {i+1}: 原始 p 值 = {p_value:.4f}, 校正后的 p 值 = {corrected_p:.4f}")
    if rejected[i]:
        print(f"  检验 {i+1} 显著（拒绝原假设）")
    else:
        print(f"  检验 {i+1} 不显著（不能拒绝原假设）")
