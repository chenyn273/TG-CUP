import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
from statsmodels.stats.multitest import multipletests
from cliffs_delta import cliffs_delta


def cliff_delta_manual(x, y):
    num_greater = 0  # x[i] > y[j]
    num_less = 0     # x[i] < y[j]
    num_equal = 0    # x[i] == y[j]


    for i in x:
        for j in y:
            if i > j:
                num_greater += 1
            elif i < j:
                num_less += 1
            else:
                num_equal += 1

    # 计算Cliff's Delta
    delta = (num_greater - num_less) / (num_greater + num_less + num_equal)
    return delta



def read_data(file_path):
    data = pd.read_csv(file_path, header=None, sep=',')
    return data.iloc[0].values  



p_values = []
for (d1, d2) in [(data1, data2), (data1, data3), (data1, data4)]:
    statistic, p_value = wilcoxon(d1, d2)
    p_values.append(p_value)

print("原始 p 值:", p_values)

rejected, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

print("校正后的 p 值:", p_values_corrected)

for i, (p_value, corrected_p) in enumerate(zip(p_values, p_values_corrected)):
    print(f"检验 {i+1}: 原始 p 值 = {p_value:.4f}, 校正后的 p 值 = {corrected_p:.4f}")
    if rejected[i]:
        print(f"  检验 {i+1} 显著（拒绝原假设）")
    else:
        print(f"  检验 {i+1} 不显著（不能拒绝原假设）")


for (d1, d2) in [(data1, data2), (data1, data3), (data1, data4)]:

    d, res = cliffs_delta(d1, d2)

    print(f"Cliff's Delta: {d},", res)
