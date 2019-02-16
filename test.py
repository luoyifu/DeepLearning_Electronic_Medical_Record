
#-*- coding: UTF-8 -*- 

'''
import time
import math
import os
import sys
import os, os.path,shutil
import codecs 


#以下函数用于提取rxr文件名
def txttq(dirname):
    import os
    import glob
    import sys
    filter = [".txt"] #设置过滤后的文件类型 当然可以设置多个类型
    result = []#所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):# print("1:",maindir) #当前主目录#print("2:",subdir) #当前主目录下的所有目录# print("3:",file_name_list) #当前主目录下的所有文件
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)#合并成一个完整路径
            ext = os.path.splitext(apath)[1] # 获取文件后缀 [0]获取的是除了文件名以外的内容
            if ext in filter:
                result.append(apath)
    return result
#print(txttq("D:\DeepLearning ER"))



txtps = txttq( u'C:/workspace/research/EMR_Database/患者病历文本语料仓库_test_database') #txt目录提取
zljhs = []
for txtp in txtps:
#txtp=txtp.decode('utf-8')
    f=open(txtp,'r',)
    for line in f.readlines():
        if line.find ('辅助检查') >-1:
            zljhs.append(line)

print(zljhs)
'''
import numpy as np 
m = np.mat(np.zeros((3,3)))
print(m[0,0])