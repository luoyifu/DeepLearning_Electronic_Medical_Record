#Auther HJ
import  os
import shutil
os.getcwd()
dir = "D:\python\\11"
dir = dir+"\\22"
print(dir)
os.chdir(dir)
ID='aa'
henggang='//'
Txt='.txt'
e= open(dir+henggang+ID+Txt,'a+')
lines = e.readlines()
print(lines)
print(os.getcwd())
oldname= "D:\\Python Project\\111"
newname="D:\\Python Project\\222"
shutil.copyfile(oldname, newname)