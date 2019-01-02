# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:14:46 2018

文本预处理函数集

清洗文本函数集合，包括清理特殊符号，过滤停止词，建立停止词列表

@author: Luo Yifu

"""

# 清洗数据，除去多余的空格以及各种符号和编码
def washdata(text_str):
    import re
    text_str = re.sub(r'[\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+', '', text_str)
    text_str = re.sub(r'\s+', '', text_str)  # trans 多空格 to空格
    text_str = re.sub(r'\n+', '', text_str)  # trans 换行 to空格
    text_str = re.sub(r'\t+', '', text_str)  # trans Tab to空格
    return text_str

# 清洗检验数据，除去各种符号和编码，对检验结果特殊符号进行替代
def wash_exam_data(text_str):
    import re
    text_str = re.sub(r'↑', '升高', text_str)
    text_str = re.sub(r'↓', '降低', text_str)
    # 如检验结果为‘肌酐62.00umol/L’，替换为‘肌酐正常’，如为‘尿酸206.00umol/L↓’替换为‘尿酸降低’
    text_str = re.sub(r'\d.*?L|\d.*?%','正常', text_str) # 匹配如2.3mol/L,2.3%等检验数值，并用‘正常’替代
    text_str = re.sub(r'正常升高','升高',text_str)
    text_str = re.sub(r'正常降低','降低',text_str)
    text_str = re.sub(r'[\s+\.\!\/_,$%*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&（）]+', '', text_str)
    text_str = re.sub(r'\s+', '', text_str)  # trans 多空格 to空格
    text_str = re.sub(r'\n+', '', text_str)  # trans 换行 to空格
    text_str = re.sub(r'\t+', '', text_str)  # trans Tab to空格
    text_str = re.sub(r'主诉|辅助检查|实验室及辅助检查|：|入院后行|入院后|我院门诊|我院急诊|入科后|我院|门诊行|请随诊|考虑|请|今日','',text_str)
    return text_str


# 清洗掉日期信息
def wash_data_info(text_str):
    import re
    text_str = re.sub(r'(\d{8}|\d{7}|\d{6})','',text_str)
    text_str = re.sub(r'(\d{4}-\d{1,2}-\d{1,2})','', text_str) # 清洗如2018-09-01样的日期信息
    text_str = re.sub(r'(\d{4}年)','', text_str)
    text_str = re.sub(r'(\d{1,2}月)','', text_str)
    text_str = re.sub(r'(\d{1,2}日)','', text_str)
    text_str = re.sub(r'(\d{1,2}:\d{1,2})','', text_str) # 清洗如9:40样的时间信息 
    return text_str

# 对于类似“行螺旋检查提示双肺炎性病变”/“心电图检查提示房颤”类型表述的信息提取
# 我们希望提取特征，即提取检查后的结果
# 以“提示/示”为关键词，查找表述中是否有该关键词，如果有，提取结果信息
def Special_Pattern_info(text_str):
    import re
    matchObj = re.search(r'(.*)[提示|示|](.*)',text_str)
    if matchObj:
        text_str = matchObj.group(2)
    else:
        text_str = text_str
    # match2 匹配形如“血氧饱和度608升高”，替换为“血氧饱和度升高”
    matchObj2 = re.search(r'(.*)[\d+][升高|降低|正常]',text_str)
    if matchObj2:
        text_str = re.sub(r'(\d+)','',text_str)
    return text_str


# 过滤停止词，并形成一个列表文件，每一行存储一个词
# 词典格式：一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒
def wash_stopword(text_data, stopwords, **dict_file_name):
    import jieba
    jieba.load_userdict(dict_file_name)  # file_name 为文件类对象或自定义词典的路径
    text_list = jieba.lcut(text_data)
    word_list = []
    for sentence in text_list:
        if sentence not in stopwords:
            word_list.append(sentence)
    return word_list


# 建立停止词列表
def stopwordslist(filepath):
    import io
    stopwords = [line.strip() for line in io.open(
        filepath, 'r', encoding='gbk').readlines()]
    return stopwords


# 如何处理无/否认的信息
# 识别否定关键词“无/否认/正常”，如果存在该关键词，删除该关键词所在条目
def deny_rules(text_str):
    import re
    str_pattern = r'无|否认|正常'
    pattern = re.compile(str_pattern)
    if pattern.search(text_str):
        return False
    else:
        return True
