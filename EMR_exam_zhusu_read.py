# -*- coding: utf-8 -*-
'''
处理电子病历中“辅助检查&主诉”部分
提取主诉和辅助检查的特征，将其向量化
每个患者表述成一个numpy array的向量，以便下一步对患者病历进行进一步分析

method:
病历数据读取，保存为list
对所有病例包含的特征短语进行非重复特征提取
患者病历向量化——词袋法（比照非重复特征）

however
患者病历向量化后，向量太稀疏，特征之间是否有关联，什么样的关联？

last edited at:
2019-01-02 Luo Yifu
'''

# numpy包引入用来构建向量和向量计算
import numpy as np
import Text_Treatment
import Pre_Treatment
import EMR_read_FeatureAbstract as erf

# ---------------------------文件准备---------------------------#
name1='辅助检查'
name2='主 诉'
name3='最后诊断'
# 病历数据库所在文件路径
file_path = (
    u'C:/workspace/research/EMR_Database/患者病历文本语料仓库_full_database')
# 处理后的数据存储文件路径
filepath_after_treatment_text_database = (
    u'C:/workspace/research/EMR_Database/temp/患者病历处理后的文本语料仓库_database/')

# 获取文件名称列表
filelist_exam = Text_Treatment.get_full_filename(file_path,name1)
filelist_zhusu = Text_Treatment.get_full_filename(file_path,name2)
filelist_zhenduan = Text_Treatment.get_full_filename(file_path,name3)

# ----------------------------数据读取-----------------------------#
# 用list存储读取到的病历数据，每个特征短语存储为list的一个元素
word_list_exam = []
word_list_zhusu = []
word_list_zhenduan = []

# 读取’辅助检查‘信息
for x in filelist_exam:
    word_list_exam.append(erf.read_exam_file_single(x))
# 读取’主诉‘信息
for x in filelist_zhusu:
    word_list_zhusu.append(erf.read_exam_file_single(x))
# 读取’最后诊断‘信息
for x in filelist_zhenduan:
    word_list_zhenduan.append(erf.read_zhenduan_file(x))

# 将2个词汇列表合并为1个
# 每个病历的主诉和检验结果都存放在一个list中，即word_list_patient


word_list_patient = word_list_exam
for i in range(len(filelist_exam)):
    word_list_patient[i][len(word_list_patient[i]):len(word_list_patient[i])] = word_list_zhusu[i]
    # 将患者列表存储到文件中，一个患者一个文件
    temp =  [filepath_after_treatment_text_database, 'treated_', str(i), '.txt']
    filename_temp = ''.join(temp)
    Text_Treatment.write_word_list_to_file(word_list_patient[i],filename_temp)

# -----------------------------非重复特征提取------------------------------#
# 提取患者病历中所有非重复特征
# 对word_list_patient进行非重复特征提取，提取特征存入exam_feature_list
exam_feature_list = []
for x in word_list_patient:
    exam_feature_list = erf.find_unique_feature(exam_feature_list, x)

zhenduan_feature_list = []
for x in word_list_zhenduan:
    zhenduan_feature_list = erf.find_unique_feature(zhenduan_feature_list, x)

# -------------------------------tfidf计算--------------------------------#
# 计算exam_feature_list中每个特征的idf值
feature_idf_list = erf.feature_idf(exam_feature_list, word_list_patient) 
# 特征进行词袋模型转化，转变成向量
feature_EMR = []
# 特征向量计算tfidf值
feature_EMR_tfidf = []
for x in word_list_patient:
    feature_EMR.append(erf.feature_EMR_exam_zhusu(exam_feature_list, x))
    feature_EMR_tfidf.append(erf.feature_tfidf(erf.feature_EMR_exam_zhusu(exam_feature_list, x), feature_idf_list))

# 计算zhenduan_feature_list中每个特征的idf值
zhenduan_idf_list = erf.feature_idf(zhenduan_feature_list, word_list_zhenduan)
# 特征进行词袋模型转化，转变成向量 
zhenduan_EMR = []
# 特征向量计算tfidf值
zhenduan_EMR_tfidf = []
for x in word_list_zhenduan:
    zhenduan_EMR.append(erf.feature_EMR_exam_zhusu(zhenduan_feature_list, x))
    zhenduan_EMR_tfidf.append(erf.feature_tfidf(erf.feature_EMR_exam_zhusu(zhenduan_feature_list, x), zhenduan_idf_list))

# -------------------------------特征向量转换成numpy array型------------------------------#
feature_EMR_np_array = np.array(feature_EMR)
feature_EMR_tfidf_np_array = np.array(feature_EMR_tfidf)
zhenduan_EMR_np_array = np.array(zhenduan_EMR)
zhenduan_EMR_tfidf_np_array = np.array(zhenduan_EMR_tfidf)

np.save('feature_EMR_np_array.npy',feature_EMR_np_array)
np.save('feature_EMR_tfidf_np_array.npy',feature_EMR_tfidf_np_array)
np.save('zhenduan_EMR_np_array.npy',zhenduan_EMR_np_array)
np.save('zhenduan_EMR_tfidf_np_array.npy',zhenduan_EMR_tfidf_np_array)

print(physical_exam_feature_list)


'''
【结果分析】：

提取主诉和检验数据
病历总数 N
得到exam_feature_list是8922行，即共有8922个特征。计为 M

ps:2019-01-03：在改进了特征提取后，特征总数降低至7417个
增加了Pre_Treatment.Special_Pattern_info函数

做法是：将所有病历中非重复的短语表述提取出来，成为一个特征列表（exam_feature_list）。

每一个病历数据，对照这个特征列表，用词袋法标注出病历特征向量，存储在feature_EMR中，feature_EMR是一个 N*M的list的list
将feature_EMR转换成numpy array的格式，即为feature_EMR_np_array
计算tf-idf值，得到feature_EMR_tfidf和同样方法numpy array化得到feature_tfidf_np_array


提取诊断数据特征
zhenduan_feature_list共6780行，即6780个诊断短语。但是，这些诊断结果有很多是相似的，只是表述方式不同。

ps:2019-01-03：在改进了特征提取后，诊断特征总数降低至6419个
增加了Pre_Treatment.Special_Pattern_info函数

与提取主诉和检验数据特征类似的，提取诊断特征。zhenduan_EMR

因此，一个问题是如何度量诊断结果的相似性，或者用什么方式将不同的表述归化为统一表述。

我们面临的一个主要问题是，特征向量，或者所有病例特征所形成的特征矩阵太稀疏。如何去稀疏化？

'''