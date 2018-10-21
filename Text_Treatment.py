# -*- coding: utf-8 -*-
"""
Created on 2018 10 06 2018
包含所有对文本文件处理的函数

@author: Luo Yifu

"""

import os

# ---------批量获取目录下文件的文件名和文件的全目录--------#

# 获取path目录下，filetype类型的所有文件的名称，写入name列表，
# 结果是'00201623_1主 诉.txt'
# filetype也可以是文件名中的关键词。例如“主诉”
# 则返回path目录下所有带有“主诉”关键词的文件
def get_filename(path, filetype):
    name = []
    for root, dirs, files in os.walk(path, True):# True表示跟随文件中的目录继续遍历
        for i in files:
            if filetype in i:
                name.append(i)
                #name.append(i.replace(filetype, ''))
    return name


# 获取完整的文件名称路径，结果存储在filename列表中，
# 结果是‘C:\Users\Administrator\Desktop\神经网络与医疗数据\test_data\00201623_1\00201623_1主 诉.txt’
def get_full_filename(path, filetype):
    filename = []
    for root, dirs, files in os. walk(path, True):
        for i in files:
            if filetype in i:
                # 新版本
                templist = [root, i]
                # 老版本
                #templist = [path, i]
                tempname = '/'.join(templist)
                tempname = tempname.replace('\\','/')
                filename.append(tempname)
    return filename


# 查看某个path目录下，带有“name”的文件，输出该文件的完整路径
# special_filename结果如
# “C:/Users/Administrator/Desktop/神经网络与医疗数据/test_data/00201623_100201623_1主 诉.txt”
def get_special_filename(path, name):
    for root, dirs, files in os. walk(path):
        for i in files:
            if name in i:
                temp = [path, i]
                special_filename = '/'.join(temp)
    return special_filename


#-------------------写入文件-------------------#

# 函数读取text_str，过滤掉stopwords, 分词并标注词性，将结果写入filename文件中
# 格式为每个词占一行，词 词性
def write_word_to_file(text_str, stopwords, filename):
    import jieba.posseg as pseg
    with open(filename, 'w+') as f:
        for x in pseg.cut(text_str):
            if x.word not in stopwords:
                f.write('{0}\t{1}\n'.format(x.word, x.flag))


# 将text_str文件分词后得到的word_list列表，写入文件filename
# 格式为每行一个词
def write_word_list_to_file(word_list, filename):
    with open(filename, 'w+') as f:
        for x in word_list:
            f.write('{0}\n'.format(x))

#--------------------读取文件---------------------------#

# 读取进行过分词和划分词性的filename，将数据导入word_list中
def read_cut_result(filename):
    word_list = []
    with open(filename, 'r') as f:
        for x in f.readlines():
            p = x.split()
            word_list.append((p[0], p[1]))
    return word_list


# 读取存入文件中的word_list
def read_word_list(filename):
    word_list = []
    with open(filename, 'r') as f:
        word_list = f.read().splitlines()
    return word_list