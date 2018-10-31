# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:14:46 2018

使用LDA方法，分析文本库中的文档

@author: luo yifu
"""
from gensim import corpora, models
import Text_Treatment
import pprint as pp

file_path = (
    u'C:/workspace/research/EMR_Database/患者病历文本语料仓库_test_database')

# 定义处理后的语料仓库地址
filepath_after_treatment_text_database = (
    u'C:/workspace/research/EMR_Database/temp/患者病历处理后的文本语料仓库_database/')

'''
filepath_after_treatment_text_database = (
    u'C:/workspace/research/EMR_Database/处理后的主诉及病史/database_treated/')
'''
# 获取语料仓库中所有.txt文件的名称
filenames_full_path = Text_Treatment.get_full_filename(
    filepath_after_treatment_text_database, '.txt')

# 每个文本存储为一个list（每个特征占一行），text_word_list是所有文本list的list
text_word_list = []
for i in filenames_full_path:
    text_word_list.append(Text_Treatment.read_word_list(i))

# 建立语料特征索引词典，并将文本特征的原始表达转化成词袋模型对应的稀疏向量表达。
dictionary = corpora.Dictionary(text_word_list)

# 读取存储的词典
# dictionary = corpora.Dictionary.load_from_text(u'C:/workspace/research/EMR_Database/temp/MedicalRecord.dict')
dictionary.save(
    u'C:/workspace/research/EMR_Database/temp/MedicalRecord.dict')  # 将词典存储以备未来使用

# 构建gensim语料仓库corpus，corpus是基于词袋模型
corpus = []
for text in text_word_list:
    corpus.append(dictionary.doc2bow(text))

# 将corpus存储到文件中以备未来使用
corpora.MmCorpus.serialize(
    u'C:/workspace/research/EMR_Database/temp/MecicalRecord.mm', corpus)  

# 读取文件中的corpus
# corpus = corpora.MmCorpus(/tmp/MecicalRecord.mm')


# tfidf&lsi模型
tfidf_model = models.TfidfModel(corpus)

# 将语料库中所有文档的词转换成tfidf模式
# 相比原本的corpus，tfidf模式的语料库对于词语的重要性有更多的强调。 
corpus_tfidf = tfidf_model[corpus]
'''
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
corpus_lsi = lsi_model[corpus_tfidf]

'''

# num_topics: 必须。LDA 模型要求用户决定应该生成多少个主题。由于我们的文档集很小，所以我们只生成15个主题。
# id2word：必须。LdaModel 类要求我们之前的 dictionary 把 id 都映射成为字符串。
# passes：可选。模型遍历语料库的次数。遍历的次数越多，模型越精确。但是对于非常大的语料库，遍历太多次会花费很长的时间
EMR_ldamodel = models.ldamodel.LdaModel(corpus_tfidf, num_topics=50, id2word=dictionary, passes=30)
 
# 将LDA模型存储到文件中
lda_model_filename = u'C:/workspace/research/EMR_Database/temp/MedicalRecord_lda.model'
EMR_ldamodel.save(lda_model_filename)

# 将所有文档所化为的主题存储在doc_topic中
# 这行代码可以将新的文档转化为LDA主题分布
doc_topic = [a for a in EMR_ldamodel[corpus]]

t = EMR_ldamodel.print_topics(50)

pp.pprint(EMR_ldamodel.print_topics(5))


# 读取已有的LDA model，输入LDA model的文件路径
def Reload_Existing_Corpus_Model(lda_model_filename):
    ldamodel = models.ldamodel.LdaModel.load(lda_model_filename)
    return ldamodel


# 输入需要建模的文字text_doc，字典dic，和lad模型的文件lda_model_filename
def LDA_Predict(text_doc, Dict, lda_model_filename):
    lda_model = Reload_Existing_Corpus_Model(lda_model_filename)
    # 文档转换成bow
    doc_bow = Dict.doc2bow(text_doc)
    # 得到新文档的主题分布
    doc_lda = lda_model[doc_bow]
    pp.pprint(doc_lda)


for t in corpus:
    EMR_ldamodel.get_document_topics(t)

doc_bow2 = dictionary.doc2bow(text_word_list[300])
doc_bow4 = dictionary.doc2bow(text_word_list[500])

doc2_lda = EMR_ldamodel[doc_bow2]
doc4_lda = EMR_ldamodel[doc_bow4]

for x in text_word_list:
    LDA_Predict(x, dictionary, lda_model_filename)
