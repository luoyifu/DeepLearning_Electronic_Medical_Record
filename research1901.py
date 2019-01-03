# -*- coding: utf-8 -*-
'''
19年研究round 1

主要针对主诉，辅助检查和最终诊断做分析

method:
1. 
计算病历之间的cos相似性，与诊断结果之间的cos相似性，
比较两组相似性，差值按照每0.1一个区间进行统计

'''
import numpy as np
from matplotlib import pyplot as plt


# ------------------------数据准备---------------------------------#
# 读取数据
# 数据来院EMR_exam_zhusu_read.py
feature_EMR_np_array = np.load('feature_EMR_np_array.npy')
feature_EMR_tfidf_np_array = np.load('feature_EMR_tfidf_np_array.npy')
zhenduan_EMR_np_array = np.load('zhenduan_EMR_np_array.npy')
zhenduan_EMR_tfidf_np_array = np.load('zhenduan_EMR_tfidf_np_array.npy')


# -------------------------计算所需函数------------------------------#
# 计算两个向量的余弦相似性函数
def cos_sim(array1,array2):
    num = float(array1 .dot(array2)) 
    denom = np.linalg.norm(array1) * np.linalg.norm(array2)
    cos = num / denom #余弦值
    sim = 0.5 + 0.5 * cos #归一化
    return sim

# 绘制直方图函数。
# 需要输入numpy array数据，绘制共20个区间的直方图
def show_histogram(data):
    n, bins, patches = plt.hist(data, 20,normed=0, edgecolor="black")
    plt.xlabel("diff range")
    # 显示纵轴标签
    plt.ylabel("frequency")
    plt.title("diff between feature sim and zhenduan sim")
    plt.show()

# -----------------------------分析-----------------------------------#

# method 1.实现
diff = []
for i in range(len(zhenduan_EMR_np_array)):
    for j in range(len(zhenduan_EMR_np_array)):
        if i !=j:
            cos_feature = cos_sim(feature_EMR_np_array[i],feature_EMR_np_array[j])
            cos_zhenduan = cos_sim(zhenduan_EMR_np_array[i],zhenduan_EMR_np_array[j])
            diff.append(abs(cos_feature - cos_zhenduan))
            if abs(cos_feature - cos_zhenduan) == 0:
                print(i,j)

# diff数据转换成numpy array
diff_array = np.array(diff)

show_histogram(diff_array)
