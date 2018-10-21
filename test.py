# -*- coding: utf-8 -*-
import Text_Treatment

import EMR_exam_zhusu_read

path = (
    u'C:/wokspace/科研&探索/EMR_Database/患者病历文本语料仓库_test_database')

name= '辅助检查'
file_list = Text_Treatment.get_full_filename(path,name)

word_list_test = []
word_list_test.append(read_exam_file_1(file_list[1]))
print(file_list)