# -*- coding: utf-8 -*-
'''
处理电子病历中“辅助检查&主诉”部分

2018-10-8 Luo Yifu
'''

# numpy包引入用来构建向量和向量计算
import numpy as np
import Text_Treatment
import Pre_Treatment

# 读取单一检查数据，并根据“。，”进行分割。分割成的每项特征写入word_list列表中
def read_exam_file_single(filename):
    import re
    word_list = []
    f = open(filename,'r')
    t = f.read()
    t = re.sub(r'\s+', '', t)  # 去除多空格 to空格
    t = re.sub(r'\n+', '', t)
    for x in re.split(r'[、，。,]', t):  # 按照“。，”对字符串进行切割
        x = Pre_Treatment.wash_data_info(x)
        x = Pre_Treatment.wash_exam_data(x)
        word_list.append(x)
    return word_list

# 读取诊断结果，并根据“。，”进行分割。分割成的每项特征写入word_list列表中
def read_zhenduan_file(filename):
    import re
    word_list = []
    f = open(filename,'r')
    t = f.read()
    t = re.sub(r'\s+', '', t)  # 去除多空格 to空格
    t = re.sub(r'\n+', '', t)
    for x in re.split(r'[、，。,]|[提示|示|]', t):  # 按照“。，”对字符串进行切割
        x = Pre_Treatment.wash_data_info(x)
        x = Pre_Treatment.wash_exam_data(x)
        x = x.lstrip('1234567890')
        word_list.append(x)
    return word_list


# --------提取病历中非重复特征--------
# 输入wordlist列表，如果列表中有元素并未在featrue_list列表中出现，
# 则在feature_list列表中添加该元素
# 最终返回feature_list列表
def find_unique_feature(feature_list, wordlist):
    for x in wordlist:
        if x not in feature_list:
            feature_list.append(x)
    return feature_list


# --------提取某个病历的特征--------
# 提取某个病历(包含辅助检查&主诉)的特征，表述为一个0,1值的向量
# 输入feature_list(总病历特征)列表，如果该病历有某个特征，则该特征值记为1，否则记为0
# wordlist为某个病例。是经过分割后的词列表
# 返回值feature_EMR是一个numpy array(向量)
def feature_EMR_exam_zhusu(feature_list, wordlist):
    feature_EMR = []
    for x in feature_list:
        if x in wordlist:
            feature_EMR.append('1')
        else:
            feature_EMR.append('0')
    feature_EMR = np.array(feature_EMR, dtype='int')
    return feature_EMR


# tf-idf方法

# 计算特征的idf值
# 输入所有文档的特征向量，wordlist是列表的列表
# featrue_list是所有特征的向量
# 输出feature_idf_list是每个特征的idf值
def feature_idf(feature_list, wordlist):
    n = len(wordlist)  # 文档总数
    feature_idf_list = []
    for i in feature_list:
        count = 0
        for j in wordlist:
            if i in j:
                count = count + 1
        t = np.log10(n/count)
        feature_idf_list.append(t)
    feature_idf_list = np.array(feature_idf_list, dtype='float')
    return feature_idf_list


# 计算某个病历的每个特征的tfidf值

# 输入某个病历的特征向量wordlist
# 输出该病历每个特征的tfidf值
def feature_tfidf(wordlist, feature_idf_list):
    nl = len(wordlist)
    feature_tfidf = np.zeros(nl)
    n_word = np.sum(wordlist)
    wordlist_t = wordlist/n_word
    for num in range(nl):
        feature_tfidf[num] = wordlist_t[num]*feature_idf_list[num]
    return feature_tfidf

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
    word_list_exam.append(read_exam_file_single(x))

for x in filelist_zhusu:
    word_list_zhusu.append(read_exam_file_single(x))

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
    exam_feature_list = find_unique_feature(exam_feature_list, x)

# 计算physical_exam_feature_list中每个特征的idf值
feature_idf_list = feature_idf(exam_feature_list, word_list_patient) 

feature_EMR = []
feature_EMR_tfidf = []
for x in word_list_patient:
    feature_EMR.append(feature_EMR_exam_zhusu(exam_feature_list, x))
    feature_EMR_tfidf.append(feature_tfidf(feature_EMR_exam_zhusu(exam_feature_list, x), feature_idf_list))

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

zhenduan_idf_list = feature_idf(zhenduan_feature_list, word_list_zhenduan) 

zhenduan_EMR = []
zhenduan_EMR_tfidf = []
for x in word_list_zhenduan:
    zhenduan_EMR.append(feature_EMR_exam_zhusu(zhenduan_feature_list, x))
    zhenduan_EMR_tfidf.append(feature_tfidf(feature_EMR_exam_zhusu(zhenduan_feature_list, x), zhenduan_idf_list))

zhenduan_EMR_np_array = np.array(zhenduan_EMR)
zhenduan_EMR_tfidf_np_array = np.array(zhenduan_EMR_tfidf)

np.save('zhenduan_EMR_np_array.npy',zhenduan_EMR_np_array)
np.save('zhenduan_EMR_tfidf_np_array.npy',zhenduan_EMR_tfidf_np_array)

print(physical_exam_feature_list)
