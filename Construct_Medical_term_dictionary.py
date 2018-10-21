# coding=utf-8
'''
编制术语字典
'''

import re

# 术语字典所在目录
dic_filename = (u'C:/Users/Administrator/Desktop/神经网络与医疗数据/supporting_files/Dic_Drug_Label.txt')

# 新加入术语字典的字典目录
dic_filename_new = (u'C:/Users/Administrator/Desktop/神经网络与医疗数据/supporting_files/药品名称中英文对照表.txt')


# 从filename字典中提取新的术语
def extract_terms(filename):
    word_list = []
    # 注意原始文件编码方式必须为utf-8
    raw = open(filename, encoding='utf-8', errors='ignore')
    temp_str = raw.read()
    temp_str = temp_str.encode('utf-8').decode('utf-8-sig')
    temp_str = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', ' ', temp_str)
    temp_str = re.sub(r'\s+', ' ', temp_str)  # 去除多空格 to空格
    temp_str = re.sub(r'\n+', ' ', temp_str)
    for x in re.split(' ', temp_str):
        if x != '':
            word_list.append(x)
    return word_list


# 原始的字典文件
word_list_origin = []
with open(dic_filename, 'r+', encoding='utf-8', errors='ignore') as f:
    temp_str = f.read()
    temp_str = temp_str.encode('utf-8').decode('utf-8-sig')
    temp_str = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', ' ', temp_str)
    temp_str = re.sub(r'\s+', ' ', temp_str)  # 去除多空格 to空格
    temp_str = re.sub(r'\n+', ' ', temp_str)
    for x in re.split(' ', temp_str):
        if x != '':
            word_list_origin.append(x)
    # 获取新的字典列表
    word_list = extract_terms(dic_filename_new)
    # 将不重复的新名词写入字典当中
    for x in word_list:
        if x not in word_list_origin:
            f.write('{0}\n'.format(x))
