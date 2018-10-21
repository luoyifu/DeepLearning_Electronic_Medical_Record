# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:23:58 2018

@author: Administrator
"""
# 预处理所有文本文件
# 读取文本仓库目录下的文件，进行过滤和分词（使用stop-word-list和jieba分词包），
# 然后写入新的文件，存储于新的文本仓库中（两种分词模式，一种分词和词性，另一种单独分词）

import Pre_Treatment
import Text_Treatment

# 定义语料仓库的文件路径
filepath_original_text_database = (
    u'C:/Users/Administrator/Desktop/神经网络与医疗数据/主诉及病史/')

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
filenames_out = []
filenames_out2 = []
for i in filenames:
    temp = [filepath_after_treatment_text_database, 'treated_', i, '.txt']
    temp2 = [filepath_after_treatment_text_database, 'treated_word_', i, '.txt']
    name_temp = ''.join(temp)
    name_temp2 = ''.join(temp2)
    filenames_out.append(name_temp)
    filenames_out2.append(name_temp2)

# 对文件进行读取，并将文件分词，过滤停止词，最后写入另一个文件

# 创建停用词列表

# 建立停止词列表（过滤掉的词语和符号）
stopwords = Pre_Treatment.stopwordslist(
    u'C:/Users/Administrator/Desktop/神经网络与医疗数据/python code/stopword.txt')

# 自定义字典可以更好的进行分词
dic_filename = ('')  # 输入自定义字典路径

for i in range(len(filenames)):
    print(i)
    raw = open(filenames_full_path[i])
    temp_str = raw.read()
    temp_str = Pre_Treatment.washdata(temp_str)
    word_list = Pre_Treatment.wash_stopword(temp_str, stopwords)
    read_rewrite_text.write_word_to_file(
        temp_str, stopwords, filenames_out2[i])
    read_rewrite_text.write_word_list_to_file(word_list, filenames_out[i])
