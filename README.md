# DeepLearning_Electronic_Medical_Record
deep learning method applying on electronic medical record

created by @luoyifu, @hejian

本项目旨在通过深度学习阅读电子病历信息

## 1. 背景与介绍
电子病历数据处理一直是一个难点。电子病历书写的随意性，中文语言的多义性，以及处理自然语言和提取病历数据中的特征的困难都制约了海量电子病历数据的应用。近年来，许多科学家和团队基于电子病历数据开发了一系列应用。

中文电子病历数据的处理面临的第一个问题就是中文分词。中文和英文在分词上有显著的差异，因此恰当有效的电子病历分词就十分重要。我们采取“jieba”包来进行分词操作。为了进一步提升分词准确性，人工创建stopwords列表（停止词）和dic_medical_term列表（医学术语词典）。借助这两个字典的帮助，我们对电子病历数据进行了较好的分词。

电子病历数据包含许多不同结构和不同模式的数据。例如，实验室检查结果可以表述为指标和分值的一一对应列表，方便直接作为特征提取出来。而体格检查则是医生手工书写，具有一定的随意性。主诉和病史（家族史、个人史、婚育史……）的表述则更佳混乱。
在处理自然语言病历上，我们选取LDA方法和word2vec方法。



## 2. 函数
### 2.1 文件读取与辅助函数
1. read_text_database.py
获取目录下所有文件的文件名称和完整路径

2. read_rewrite_text.py
对字符串进行分词，过滤停止词，并将结果写入文件当中
读取之前写入的分词完毕的文件

3. Pre_Treatment.py
清洗文本函数集合，包括清理特殊符号，过滤停止词，建立停止词列表

4. Construct_Medical_term_dictionary.py
编制术语字典

### 2.2 病历特征提取函数集
1. EMR_physical_exam_read.py
处理电子病历中“体格检查”部分

2. EMR_medical_history_read.py
读取主诉，个人史，既往史，家族史，婚育史信息
读取某个文件夹下所有病史数据文档
进行分词和特征提取，并将结果存储到文件中

### 2.3 文本数据分析函数
1. gensim_LDA.py
使用LDA方法，分析文本库中的文档

## 3. 其他资源
* stopword.txt
停止词列表
* stopword_MedicalRecord.txt
医学病历数据停止词列表
* Dic_Medical_Record.txt
医学术语词典
使用LDA方法，分析文本库中的文档资源
使用LDA方法，分析文本库中的文档

## 4. 其他文档介绍
* Statment_EMR_medical_history.md: 介绍处理主诉、病史等数据的思想和处理方法
