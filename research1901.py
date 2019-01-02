# -*- coding: utf-8 -*-
'''
19年研究round 1

主要针对主诉，辅助检查和最终诊断做分析

'''
import numpy as np


# ------------------------数据准备---------------------------------#
# 读取数据
# 数据来院EMR_exam_zhusu_read.py
feature_EMR_np_array = np.load('feature_EMR_np_array.npy')
feature_EMR_tfidf_np_array = np.load('feature_EMR_tfidf_np_array.npy')
zhenduan_EMR_np_array = np.load('zhenduan_EMR_np_array.npy')
zhenduan_EMR_tfidf_np_array = np.load('zhenduan_EMR_tfidf_np_array.npy')


# -------------------------计算所需函数------------------------------#
# 定义计算余弦相似性函数
def cos_sim(array1,array2):
    num = float(array1 .dot(array2)) 
    denom = np.linalg.norm(array1) * np.linalg.norm(array2)
    cos = num / denom #余弦值
    sim = 0.5 + 0.5 * cos #归一化
    return sim

c1 = cos_sim(zhenduan_EMR_np_array[0],zhenduan_EMR_np_array[1])
c2 = cos_sim(zhenduan_EMR_np_array[0],zhenduan_EMR_np_array[2])