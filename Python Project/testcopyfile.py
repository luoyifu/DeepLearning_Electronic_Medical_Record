#Auther HJ

import os
import shutil
import mkdir
pre="d:\MEDDOC\\"
#alllist=os.listdir(u"D:\\notes\\python\\资料\\")
dir='D:\MEDDOC'



def path_form(patient_id,visit_id):
    if (len(str(patient_id))==10):
        numlist = [str(patient_id[item: item + 2]) for item in range(0, len(patient_id), 2)]
        path=numlist
        path="\\".join(str(i) for i in path)
        path=pre+path+"\IP_"+visit_id
        print(path)
        return path
    if(len(str(patient_id))<10):
        patient_id=patient_id.zfill(10)
        numlist = [str(patient_id[item: item + 2]) for item in range(0, len(patient_id), 2)]
        path = numlist
        path = "\\".join(str(i) for i in path)
        path = pre+path + "\IP_" + visit_id
        print(path)
        return path
    else:
        print("错误！，Patient_id超过10位，")
def CopyFileEmr(patient_id,visit_id,title_name):
    ori_path=path_form(patient_id,visit_id)+'\\'+title_name+'.txt'
    print("ori_path:",ori_path)
    des_path=mkdir.mkdirs(patient_id,visit_id)+'/'+patient_id+'_'+visit_id+'.txt'
    print("des_path:", des_path)
    shutil.copy(ori_path,des_path)




#CopyFileEmr('90067453','1')
