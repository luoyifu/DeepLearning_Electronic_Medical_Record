# -*- coding: utf-8 -*-
'''
处理电子病历中“体格检查”部分

2018-4-2 Luo Yifu
'''

# numpy包引入用来构建向量和向量计算
import numpy as np
import EMR_read_FeatureAbstract as erf



# 测试部分

filename = 'patient_physical_exam.txt'
filename2 = 'patient_physical_exam2.txt'

word_list = []
word_list.append(erf.read_physical_exam_file(filename))
word_list.append(erf.read_physical_exam_file(filename2))

physical_exam_feature_list = []

physical_exam_feature_list = erf.find_unique_feature(
    physical_exam_feature_list, word_list[0])
physical_exam_feature_list = erf.find_unique_feature(
    physical_exam_feature_list, word_list[1])

# 计算physical_exam_feature_list中每个特征的idf值
feature_idf_list = erf.feature_idf(physical_exam_feature_list, word_list) 

feature_EMR_p1 = erf.feature_EMR_physical_exam(
    physical_exam_feature_list, word_list[0])

# 计算word_list[0]病历的每个特征的tfidf值
feature_tfidf_1 = erf.feature_tfidf(feature_EMR_p1, feature_idf_list)

print(physical_exam_feature_list)
