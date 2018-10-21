# -*- coding: utf-8 -*-
'''
读取患者相关文档，将每个患者编制成一个向量

Author: Luo Yifu
2018.6.21
'''

import numpy
import Text_Treatment
import os
import EMR_physical_exam_read

#---读取数据部分---#

# 完整语料仓库地址为：
# C:/Users/Administrator/Desktop/神经网络与医疗数据/Database/患者病例文本语料仓库_full_database/
# 该数据库每个文件夹为一个患者的相关病历
patient_dic=(
    u'C:/Users/Administrator/Desktop/神经网络与医疗数据/Database/患者病历文本语料仓库_test_database/')

# 获取患者ID(患者ID即为文件夹名)
patient_dic_id = []
for root, dir, files in os.walk(patient_dic, True):
    for i in dir:
        patient_dic_id.append(i)

# 获取数据库中所有带有关键词“X”的文件名，且返回完整文件目录
patient_dic_file=read_text_database.get_full_filename(patient_dic, '体格检查')


#---特征提取部分---#

# 患者特征会存储在向量中。向量中1表示有这个特征，0表示没有。
patients_feature_array_name = [] # 存储患者特征的向量的向量名
'''
for id in patient_dic_id:
    temp = '_'.join(id,'特征向量')
    patients_feature_array_name.append(temp)
'''


# 创建所有体格检查特征列表
physical_exam_feature_list = []

for filename in patient_dic_file:
    wordlist = EMR_physical_exam_read.read_physical_exam_file(filename)
    physical_exam_feature_list = EMR_physical_exam_read.find_unique_feature(
        physical_exam_feature_list, wordlist)


# 为每个患者分配一个特征向量
# patients_feature_array_name是一个list元素，存储所有患者的特征
# patients_feature_array_name[0](即这个List中的每个元素)都是一个numpy.ndarray元素
# 每个ndarray元素存储某个患者的特征向量

patients_feature_array_name = []
for filename in patient_dic_file:
    wordlist = EMR_physical_exam_read.read_physical_exam_file(filename)
    patients_feature_array_name.append(
        EMR_physical_exam_read.feature_EMR_physical_exam(physical_exam_feature_list, wordlist))

print(physical_exam_feature_list)