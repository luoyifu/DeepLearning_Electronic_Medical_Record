# -*- coding: utf-8 -*-
'''
19年研究round 1

主要针对主诉，辅助检查和最终诊断做分析

method:
1. 
计算病历之间的cos相似性，与诊断结果之间的cos相似性，
比较两组相似性，差值按照每0.1一个区间进行统计

2.
自然的，如果病历相似，那么诊断应该也是相似的。
选取一个病历，计算其最相似的病历，然后看相似病历的诊断与选取病历诊断的相似性

'''
import numpy as np
from matplotlib import pyplot as plt


# ------------------------数据准备---------------------------------#
# 读取数据
# 数据来院EMR_exam_zhusu_read.py
feature_EMR_np_array = np.load('feature_EMR_np_array.npy')
feature_EMR_tfidf_np_array = np.load('feature_EMR_tfidf_np_array.npy')
zhenduan_EMR_np_array = np.load('zhenduan_EMR_np_array.npy')
zhenduan_EMR_tfidf_np_array = np.load('zhenduan_EMR_tfidf_np_array.npy')


# -------------------------计算所需函数------------------------------#
# 计算两个向量的余弦相似性函数
def cos_sim(array1,array2):
    num = float(array1 .dot(array2)) 
    denom = np.linalg.norm(array1) * np.linalg.norm(array2)
    cos = num / denom #余弦值
    sim = 0.5 + 0.5 * cos #归一化
    return sim

# 绘制直方图函数。
# 需要输入numpy array数据，绘制共20个区间的直方图
def show_histogram(data):
    n, bins, patches = plt.hist(data, 30, normed=1, edgecolor="black")
    plt.xlabel("diff range")
    # 显示纵轴标签
    plt.ylabel("frequency")
    plt.title("diff between feature sim and zhenduan sim")
    plt.show()

def write_to_excel(data):
    import pandas as pd
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter('Save_Excel.xlsx')
    data_df.to_excel(writer,'page_1') # float_format 控制精度
    writer.save()

''' 
# change the index and column name
data_df.columns = ['A','B','C','D','E','F','G','H','I','J']
data_df.index = ['a','b','c','d','e','f','g','h','i','j']
 '''


# -----------------------------分析-----------------------------------#

# method 1.实现
# 病历相似性和诊断相似性的差
diff = []
l = len(feature_EMR_tfidf_np_array)
cos_sim_mat_feature = np.mat(np.zeros((l,l)))
cos_sim_mat_zhenduan = np.mat(np.zeros((l,l)))
for i in range(len(zhenduan_EMR_tfidf_np_array)):
    for j in range(len(zhenduan_EMR_tfidf_np_array)):
        if i !=j:
            cos_feature = cos_sim(feature_EMR_tfidf_np_array[i],feature_EMR_tfidf_np_array[j])
            cos_sim_mat_feature[i,j] = cos_feature

            cos_zhenduan = cos_sim(zhenduan_EMR_tfidf_np_array[i],zhenduan_EMR_tfidf_np_array[j])
            cos_sim_mat_zhenduan[i,j]= cos_zhenduan
            diff.append(cos_feature - cos_zhenduan)
            
            if abs(cos_feature - cos_zhenduan) == 0:
               print(i,j)
        else:
            cos_sim_mat_feature[i,j] = 0
            cos_sim_mat_zhenduan[i,j] = 0

# diff数据转换成numpy array

diff_array = np.array(diff)
np.save('diff_tfidf_array',diff_array)
np.save('cos_sim_mat_feature_tfidf',cos_sim_mat_feature)
np.save('cos_sim_mat_zhenduan_tfidf',cos_sim_mat_zhenduan)

# diff_array = np.load('diff_array.npy')

# 把histogram结果写入excel
n, bins, patches = plt.hist(diff_array, 30,normed=0, edgecolor="black")
data = np.vstack((bins[1:31],n))
write_to_excel(data)

show_histogram(diff_array)
print("all done")

# -----------------------------分析-----------------------------------#
'''
# method 2.实现
cos_sim_mat_feature = np.load('cos_sim_mat_feature_tfidf.npy')
cos_sim_mat_zhenduan = np.load('cos_sim_mat_zhenduan_tfidf.npy')
l = len(feature_EMR_np_array)


# 将cos_sim_mat_feature对角线元素设为1
#for i in range(l):
#    for j in range(l):
#        if i == j:
#            cos_sim_mat_feature[i,j] = 1


# 预测的诊断和实际诊断之间的差异
diff_predict = []

index_max = []
# 根据病历相似性，选取最相似的病历，即为temp_feature，
# 然后，计算该病历与temp_feature编号病历之间诊断相似性，即为diff_predict
for i in range(l):
    temp_feature = np.argmax(cos_sim_mat_feature[i,:])
    #temp = np.where(cos_sim_mat_feature[i,:] == min(cos_sim_mat_feature[i,:]))
    index_max.append(temp_feature)
    # 
    diff_predict.append(cos_sim_mat_zhenduan[i,temp_feature])

diff_predict_array = np.array(diff_predict)

# 预测结果作图显示
n, bins, patches = plt.hist(diff_predict, 30,normed=0, edgecolor="black")
show_histogram(diff_predict)
data = np.vstack((bins[1:31],n))
write_to_excel(data)

print(index_min)
'''