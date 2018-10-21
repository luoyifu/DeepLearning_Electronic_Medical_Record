#Auther HJ
import  re
import  os

def decompose(path,title_name,patientid,id):
    #查找到文件夹最后一个文件名（一般是入院记录）
    #list1=os.listdir('E:/罗一夫课题/DeepLearning ER/00202736_2')
    #把病人ID号、时间按list列出来，取最后一个即入院记录
   # res = re.findall(r'\d+(?#\D)',list1[-1])
    #print("E:/罗一夫课题/DeepLearning ER/00202736_2/"+list1[-1])
    #patient_id+住院次数
    #id=res[0]+'_'+res[-1]
    #文件名
    name1='主 诉：'
    name2='体格检查：'
    name3='最后诊断：'
    name4= '辅助检查：'
    name5='基本信息：'
    name6='现病史：'
    name7='既往史：'
    name8='个人史：'
    name9='婚育史：'
    name10='家族史：'
    id=str(patientid+'_'+id)
    filename1=id+name1
    filename2=id+name2
    filename3=id+name3
    filename4=id+name4
    filename5=id+name5
    filename6=id+name6
    filename7=id+name7
    filename8=id+name8
    filename9=id+name9
    filename10=id+name10

    #f= open("E:/罗一夫课题/DeepLearning ER/00202736_2/"+list1[-1])
    url=path+title_name+'.txt'
    f = open(url)
    g= open(path+filename1+".txt",'w')
    h= open(path+filename2+".txt",'w')
    i= open(path+filename3+".txt",'w+')
    j= open(path+filename4+".txt",'w+')
    l= open(path+filename5+".txt",'w+')
    m= open(path+filename6+".txt",'w')
    n= open(path+filename7+".txt",'w')
    o= open(path+filename8+".txt",'w')
    p= open(path+filename9+".txt",'w')
    q= open(path+filename10+".txt",'w')
    #读出各部分进行分解
    lines = f.readlines()
    for line in lines:
        if name1 in line:
            str1 = str(line.split(','))
            str1 = str1.replace('\\n', '')
            g.write(str1)
        if name2 in line:
            str2 = str(line.split(','))
            h.write(str2)
        if name4 in line:
            str4 = str(line.split(','))
            j.write(str4)
        if name6 in line:
            str6 = str(line.split(','))
            m.write(str6)
        if name7 in line:
            str7 = str(line.split(','))
            n.write(str7)
        if name8 in line:
            str8 = str(line.split(','))
            o.write(str8)
        if name9 in line:
            str9 = str(line.split(','))
            p.write(str9)
        if name10 in line:
            str10 = str(line.split(','))
            q.write(str10)
        if '1、'  in line:
            str11 = str(line.split('、'))
            i.write(str11)
        if '2、'  in line:
            str12 = str(line.split('、'))
            i.write(str12)
        if '3、'  in line:
            str13 = str(line.split('、'))
            i.write(str13)
        if '4、' in line:
            str14 = str(line.split('、'))
            i.write(str14)
        if '5、' in line:
            str15 = str(line.split('、'))
            i.write(str15)
        if '6、' in line:
            str16 = str(line.split('、'))
            i.write(str16)
        if '7、' in line:
            str17 = str(line.split('、'))
            i.write(str17)
        if '8、' in line:
            str18 = str(line.split('、'))
            i.write(str18)
        if '9、' in line:
            str19 = str(line.split('、'))
            i.write(str19)
        if '10、' in line:
            str20 = str(line.split('、'))
            i.write(str20)
        else:
            print('')
    f.close()
    g.close()
    h.close()
    m.close()
    n.close()
    o.close()
    p.close()
    q.close()
    i.close()
#去除诊断的重复字段
    k= open(path+filename3+".txt",'r+')
    txt = k.read()
    #去除特殊符号
    r='[’ ,!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    txt=re.sub(r,'',txt)
    #去除\n
    txt=txt.replace('\\n', ',')
    print(txt)
    #对诊断进行处理
    b=open(path+filename3+".txt",'w+')
    #各个诊断用逗号隔开
    txt1=txt.split(',')
    #txt1=re.split(',| |"',txt)
    #对诊断按数字进行排序
    txt1=list(set(txt1))
    txt1.sort()
    #去掉空字符串
    txt1 = [x for x in txt1 if x != '']
    b.write(str(txt1))
    b.close()
    k.close()
    #基本信息的录入
    w1 = '姓名'
    w2 = '病史陈述者'
    t= open(url)
    buff = t.read()
    pat = re.compile(w1+'(.*?)'+w2,re.S)
    result = pat.findall(buff)
    result=str(result).replace('\\n', ' ')
    print(result)
    l.write(result)
    t.close()
  #  print(line.split())
def main():
    print("hello")
    decompose('E:/罗一夫课题/DeepLearning ER/00202736_2/','00202736_34763_20171004115316_1','00202736','1')
if __name__ == '__main__':

    main()