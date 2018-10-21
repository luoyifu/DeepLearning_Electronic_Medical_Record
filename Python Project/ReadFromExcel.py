#Auther HJ
from mmap import mmap, ACCESS_READ
import xlrd
from xlrd import open_workbook
import testcopyfile
import mkdir
import htmltotxt
import dowithtext

testxls = 'D:\Python Project\数据\huxiemr2016.xls'

def readfromexcle(testxls):
    with open(testxls, 'rb') as f:
        open_workbook(file_contents=mmap(f.fileno(), 0, access=ACCESS_READ))
    wb = open_workbook(testxls)
    s= wb.sheets()[0]
    for row in range(s.nrows):
        name = s.cell(row, 1).value
        patient_id=s.cell(row, 2).value
        visit_id=s.cell(row, 3).value
        mkdir.mkdirs(patient_id,visit_id)
        path=testcopyfile.path_form(patient_id, visit_id)
        print('path:',path,'title_name:',name)
        htmltotxt.htmlTotxt(path+'\\',name)
        testcopyfile.CopyFileEmr(patient_id,visit_id,name)
        url=path+'\\'
        dowithtext.decompose(url,name,patient_id,visit_id)
        print(path)
        print(name,patient_id,visit_id)

readfromexcle(testxls)
