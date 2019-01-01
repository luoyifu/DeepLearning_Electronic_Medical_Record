# -*- coding: utf-8 -*-
'''
处理电子病历中“辅助检查&主诉”部分

2018-10-8 Luo Yifu
'''

# numpy包引入用来构建向量和向量计算
import numpy as np
import Text_Treatment
import Pre_Treatment
import EMR_read_FeatureAbstract as erf
# 测试部分

name1='辅助检查'
name2='主 诉'
name3='最后诊断'
file_path = (
    u'C:/workspace/research/EMR_Database/患者病历文本语料仓库_full_database')

filelist_exam = Text_Treatment.get_full_filename(file_path,name1)
filelist_zhusu = Text_Treatment.get_full_filename(file_path,name2)
filelist_zhenduan = Text_Treatment.get_full_filename(file_path,name3)
# word_list_test = []
# word_list_test.append(read_exam_file_single(filelist_exam[1]))

word_list_exam = []
word_list_zhusu = []

for x in filelist_exam:
    word_list_exam.append(erf.read_exam_file_single(x))

for x in filelist_zhusu:
    word_list_zhusu.append(erf.read_exam_file_single(x))

# 将2个词汇列表合并为1个
filepath_after_treatment_text_database = (
    u'C:/workspace/research/EMR_Database/temp/患者病历处理后的文本语料仓库_database/')

word_list_patient = word_list_exam
for i in range(len(filelist_exam)):
    word_list_patient[i][len(word_list_patient[i]):len(word_list_patient[i])] = word_list_zhusu[i]
    # 将患者列表存储到文件中，一个患者一个文件
    temp =  [filepath_after_treatment_text_database, 'treated_', str(i), '.txt']
    filename_temp = ''.join(temp)
    Text_Treatment.write_word_list_to_file(word_list_patient[i],filename_temp)

exam_feature_list = []
for x in word_list_patient:
    exam_feature_list = erf.find_unique_feature(exam_feature_list, x)

# 计算physical_exam_feature_list中每个特征的idf值
feature_idf_list = erf.feature_idf(exam_feature_list, word_list_patient) 

feature_EMR = []
feature_EMR_tfidf = []
for x in word_list_patient:
    feature_EMR.append(erf.feature_EMR_exam_zhusu(exam_feature_list, x))
    feature_EMR_tfidf.append(erf.feature_tfidf(erf.feature_EMR_exam_zhusu(exam_feature_list, x), feature_idf_list))

feature_EMR_np_array = np.array(feature_EMR)
feature_EMR_tfidf_np_array = np.array(feature_EMR_tfidf)

np.save('feature_EMR_np_array.npy',feature_EMR_np_array)
np.save('feature_EMR_tfidf_np_array.npy',feature_EMR_tfidf_np_array)

# 读取最后诊断信息
word_list_zhenduan = []
for x in filelist_zhenduan:
    word_list_zhenduan.append(read_zhenduan_file(x))

zhenduan_feature_list = []
for x in word_list_zhenduan:
    zhenduan_feature_list = find_unique_feature(zhenduan_feature_list, x)

zhenduan_idf_list = erf.feature_idf(zhenduan_feature_list, word_list_zhenduan) 

zhenduan_EMR = []
zhenduan_EMR_tfidf = []
for x in word_list_zhenduan:
    zhenduan_EMR.append(erf.feature_EMR_exam_zhusu(zhenduan_feature_list, x))
    zhenduan_EMR_tfidf.append(erf.feature_tfidf(erf.feature_EMR_exam_zhusu(zhenduan_feature_list, x), zhenduan_idf_list))

zhenduan_EMR_np_array = np.array(zhenduan_EMR)
zhenduan_EMR_tfidf_np_array = np.array(zhenduan_EMR_tfidf)

np.save('zhenduan_EMR_np_array.npy',zhenduan_EMR_np_array)
np.save('zhenduan_EMR_tfidf_np_array.npy',zhenduan_EMR_tfidf_np_array)

print(physical_exam_feature_list)
