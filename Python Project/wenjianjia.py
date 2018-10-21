#Auther HJ#coding:utf8

import os

def dirlist(path, allfile):
    filelist =  os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

print(dirlist("E:\罗一夫课题", []))