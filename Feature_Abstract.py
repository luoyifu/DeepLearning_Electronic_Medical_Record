'''
主诉、个人史、既往史、体格检查等都是分隔的文件
读取目录下所有文件夹中的“主诉\既往史\体格检查……”

进行分词和特征提取，将结果存储于文件中
'''

import Pre_Treatment
import Text_Treatment
import os
import re
import jieba
from Pre_Treatment import deny_rules

# 定义语料仓库的文件路径filepath_original_text_database
filepath_original_text_database = (
    u'C:/Users/Administrator/Desktop/神经网络与医疗数据/test_data/')


# 获取filepath_original_text_database目录下，所有子目录的路径
son_path = []
for root, dirs, files in os.walk(filepath_original_text_database):
        for i in dirs:
            son_path.append(i)

# filenames是数据库中，所有子目录的文件目录的合并列表
filenames = []
for i in son_path:
    filenames.append(filepath_original_text_database+i)

# 决定读取那部分数据
name1='主 诉'
name2='体格检查'
name3='最后诊断'
name4= '辅助检查'
name5='基本信息'
name6='现病史'
name7='既往史'
name8='个人史'
name9='婚育史'
name10='家族史'

temp = read_text_database.get_special_filename(filenames[0], name1)



'''
# 获取语料仓库中所有.txt文件的名称
filenames_full_path = read_text_database.get_full_filename(
    filepath_original_text_database, '.txt')

# 只获取语料仓库中文件的名称
filenames = read_text_database.get_filename(
    filepath_original_text_database, '.txt')


# 定义处理后的语料仓库地址
filepath_after_treatment_text_database = (
    u'C:/Users/Administrator/Desktop/神经网络与医疗数据/处理后的主诉及病史/')
# 定义filenames_out列表，存储输出文件名称
filenames_out = []  # 存储分词后结果
for i in filenames:
    temp = [filepath_after_treatment_text_database, 'pro_treated_', i, '.txt']
    name_temp = ''.join(temp)
    filenames_out.append(name_temp)

# 对文件进行读取，并将文件分词，过滤停止词，最后写入另一个文件

# 创建停用词列表

# 建立停止词列表（过滤掉的词语和符号）
stopwords = Pre_Treatment.stopwordslist(
    u'C:/Users/Administrator/Desktop/神经网络与医疗数据/supporting_files/stopword_MedicalRecord.txt')

# 自定义字典可以更好的进行分词
# 输入自定义字典路径
dic_filename = (u'C:/Users/Administrator/Desktop/神经网络与医疗数据/supporting_files/Dic_Medical_Record.txt')


for i in range(len(filenames)):
    jieba.load_userdict(dic_filename)
    word_list = []
    raw = open(filenames_full_path[i])
    temp_str = raw.read()
    temp_str = re.sub(r'\s+', '', temp_str)  # 去除多空格 to空格
    temp_str = re.sub(r'\n+', '', temp_str)
    for x in re.split(r'[，。：:]', temp_str):  # 按照“。，:”对字符串进行切割
        if deny_rules(x):  # 如果是否认条目，则删除该条目信息
            if x not in stopwords:
                x = jieba.lcut(x)
                for t in x:
                    if t not in stopwords:
                        word_list.append(t)
    read_rewrite_text.write_word_list_to_file(word_list, filenames_out[i])
'''