# Statement of Electronic Medical Record and Medical History Record
# 电子病历（主诉、病史等）处理的说明

---

## 数据预处理
1. 分词过程应格外考虑医学术语和药品名称术语，这些术语在中文中容易被分成其他无意义词汇。因此，分词中应用到**术语字典**。
2. 对于否认信息，我们认为是无用的。即类似“无高血压史、否认手术史、营养正常”等信息。
Pre_Treatment包：
deny_rules函数用来处理否认信息
3. 对于用药剂量，时间数据的处理？


## 给语料库中的每个词赋予不同的权重
基于权重的考量：
* 结合tfidf模型，对于某些词语利用算法决定其重要性
* **医学术语重要性加成：**对于某些医学术语，对其重要性应该加成。因为该术语可能提示更加重要的医学信息


## tfidf&lsi模型
将语料库中所有文档的词转换成tfidf模式
相比原本的corpus，tfidf模式的语料库对于词语的重要性有更多的强调。

```python
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
```

## LSI模型

```python
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
corpus_lsi = lsi_model[corpus_tfidf]
```

## 创建LDA model
num_topics: 必须。LDA 模型要求用户决定应该生成多少个主题。由于我们的文档集很小，所以我们只生成15个主题。
id2word：必须。LdaModel 类要求我们之前的 dictionary 把 id 都映射成为字符串。
passes：可选。模型遍历语料库的次数。遍历的次数越多，模型越精确。但是对于非常大的语料库，遍历太多次会花费很长的时间

```python
# 基于普通的语料库进行LDA建模
EMR_ldamodel = models.ldamodel.LdaModel(corpus, num_topics=20, id2word=dictionary, passes=30)

#基于经过tfidf变换的语料库进行LDA建模
EMR_ldamodel = models.ldamodel.LdaModel(corpus_tfidf, num_topics=20, id2word=dictionary, passes=30)
```



