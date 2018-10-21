# Research Summary of Electronic Medical Records Data
# 电子病历数据的研究摘要
---

## Introduction
主要介绍国内外在电子病历应用领域的研究内容和研究方向
精准医疗的一个主要目标就是开发出针对患者的量化模型，可以预测健康状况，帮助预防疾病，甚至辅助诊断、临床决策和治疗。临床电子病历（electronic health records, EHRs）显示了在这个领域很多可能性。[1]
电子病历的二次运用可以进行：

* 数据驱动的药物疗效和相互作用的预测[2]
* 发现二型糖尿病的子类[3]
* 发现自闭症谱系障碍合并症集群[4]
* 改善临床试验招募患者[5]

但在临床决策支持系统中没有广泛应用。

临床电子病例由于其高维度、有噪音、异质性、稀疏性、不完整性、随机错误以及系统偏性，使得临床病例的机器学习应用十分困难。[6,7]

预测算法很大程度基于特征选取和数据表示。传统的电子病历应用时找一个该领域专家制定所需寻找的特征（如学习目标和寻找目标），然后用临时方式来指定临床变量。

为了克服这些缺陷，数据驱动的电子病历特征选择方法被开发出来。这些方法的一个限制时患者通常用一个二维向量来表示，由临床数据仓库中所有数据描述组成。这种表示方式稀疏、有噪音、重复性，不适宜对电子病历中隐藏或嵌入的分层信息建模。

## 电子病历数据特征提取

电子病历是医疗机构生成的针对于医疗活动过程中文字、图表等数据的数字化信息，而且也是便于转储、管理和传输的医疗记录，其中的内容是由医务人员撰写的与患者开展医疗有关的过程实录， 包括病程记录、出院小结等部分。电子病历中包含了大量丰富的医疗知识，通过分析即可得到诸如疾病的患病特征、 用药情况以及治疗方式等各项之间的潜在联系。这样的知识数据可以对医疗问题决策提供有建设性的帮助， 并且还可以为用户建立个性化的健康模型。电子病历是结构化文本和非结构化文本相结合的一种知识数据， 因此可以通过自然语言处理的方法，来对其进行信息的抽取，以得到有用的医疗知识。电子病历中的一些专业概念，在自然语言处理问题中可称为实体，例如药品名称、治疗名称，实体和实体之间存在着语义关系，当两个实体 出现在一个句子中时， 实体以及其对应的上下文就决定了这两个实体之间的关系。实体关系抽取任务可以完成对给定实体关系类型的判断，针对于电子病历中的数据，就可以选择合理的特征来对实体之间的关系实现有效甄别。

电子病历中的实体关系抽取主要针对疾病、治疗和检查之间的关系来进行和展开，研究中选用的关系定义来源于I2B2评测提供的8种实体关系类型，例如关系TrIP定义为治疗改善或治愈了医疗问题 ［4］ 。抽取这几类实体间的关系 可以构造基于患者健康状况的个体病历的简明摘要，并且可以发现潜在的药物之间的联系。同时以医疗问题为中心，将抽取得到的实体关系组织起来，由此而形成对于医疗知识中疾病、治疗和检查等概念的系统表示。

目前电子病历中的实体关系抽取主要采取机器学习的方法，将关系抽取任务转化为多分类的问题。其具体过 程为：首先对候选实体进行特征选择，加入医疗知识作为辅助分析， 并将抽取得到的特征转化为特征向量，在向量空间模型中进行有监督学习的分类判别，由此而得到实体对的关系。鉴于电子病历的结构特殊性和领域特殊性，对于其任务中的特征选择，将在很大程度上影响关系识别的准确性。但是对于全领域中的特征选择，并没有能对领域中特征之间的关系实现整合，因而也并未形成有判别性的特征。同时，若进一步考虑到有些实体所处上下文信息不足，将医疗领域丰富的词典知识适当地引入就显得尤为必要。

电子病历中的实体关系抽取主要针对单句中的实体对， 进行关系类别判定时，主要采用的方法则是机器学习中的分类算法［4］ 。 在已有的研究中，Ｒoberts［7］ 在临床信息抽取系统CLEF 中使用了SVM分类器来实现对关系的识别，研究中试图对跨句子的关系识别进行尝试，但是得到的准确率却较低。Uzuner等［4］针对电子病历单句内实体关系的抽取，把关系识别视为分类问题，问题中的实体关系可分成6大类，并针对每一种类均训练一个特定的分类器，以实现对应关系类别的辨识，并对其特征选择了实体相对位置特征、组成实体的词及其上下文、上下文之间的依赖关系等语法。Demner－Fushman ［8］ 针对I2B22010数据，对关系抽取任务的特征选择 部分添加了其他资源以助力提升识别的准确率，再融入医疗词典资源和UMLS中的概念关系来扩充特征，来共同解决部分实体上下文内容比较稀疏的问题，研究结果表明词典资源在电子病历关系抽取任务中起到了重要作用。BerrydeBrui-jn ［9］ 则对比研究了有监督分类和基于自学习的半监督分类 方法在关系抽取中的效果表现，更通过加入未标注数据和句法分析中的依存分析结果，而使得关系抽取的识别在效果上有比较明显的提升。XiaoyanWang等 ［10］ 还对电子病历从统 计方法入手，来计算疾病和症状的上下文共现以挖掘对应实 体对之间的关联关系。OanaFrunza等［11］ 又针对疾病和治疗 之间的三种关系研究中，选择在每个多分类模型上均给出一个可能性最大的类型预测结果，而在分类模型的选择上，则在以朴素贝叶斯为代表的概率分类模型和以SVM为代表的线性分类模型上取得了较好的结果。

* 英文电子病历会遇到**缩写词**的问题
* 中文电子病历面临分词困难的问题

电子病历中词语存在着大小写不统一、词形以及领域词缩写等问题，针对这些问题，要对电子病历进行词级别的预处理，也就是进行过滤和修改。电子病历中也存在一些缩写词，示例则如图1所示，在意义上若为了明确表示， 应该将其展开为多个词的组合，也就是将缩写词进行拆分以得到对应的词，有利于更小粒度的特征提取，这样就形成更多的组合特征。还有些缩写在医疗领域中具有着特殊的意义，为此这些词也需要进行展开，实现方法是通过使用UMLS中提供的metamap叙词表来进行词的展开和替换

实体关系抽取问题的基本方法是将其转化为一个多分类问题 ［4］ ，对实体进行特征提取，转化为特征向量，再进行分 类器的训练学习。对于实体关系抽取的任务，其关系的描述多取决于词级别和上下文的词之间的关系和组合，为此将选择特征如下： 实体中包含的词、实体中包含的词的词性、实体前的2个词、实体后的2个词、实体前的2个词的词性、实体后的2个词的词性、实体的类型特征、实体之间的位置关系特征。 将这些特征组合为特征向量，词特征可表述为（wi－2，wi－1，wi+1，wi+2），对应的实体特征则表示为（type，wordendbegin），其中wordend begin包含了实体中所有的词， type为对应的实体类型。其后，则将得到的特征向量引入关系类别识别的分类任务中，即对训练数据实行有监督的学习［15］ 。 2．3基于深度学习的关系抽取2．3．1 词特征的特点 由于词特征在自然语言处理中是比较低级的特征，而对于词之间的组合和共现关系则可以得到更高级的特征，这样的特征可以表示更加丰富的意义；或者，多个词之间的组合可以与单个词有相近的语义表达，这样的特征对于实体关系识别将具有特别重要的意义，可以作为一个有判别性的特征对分类进行有针对性的指导。2．3．2 特征学习 机器学习问题分为两个部分。第一部分是对数据进行处理，并且针对研究任务选择合适的特征和表示来对数据进行形式化描述，第二部分则是针对数据的表示来进行分类模型的训练和学习，如此即使分类模型在给定的数据表示下得到可判别的效果。 对词的特征的进一步表示可以转化为特征学习的问题，问题实质是如何从数据中学习得到一个符合任务的特征表示。考虑到词之间组合的层次关系，以及概念意义的抽象层次，可选择深度学习的结构来进行多层表示的学习。


[4]UZUNEＲDSO，SOUTHBＲ，SHENS．2010I2B2/VAchallenge onconcepts，assertions，andrelationsinclinicaltext．Challenge，2011，18（5）：552–557.
[5］张奇．信息抽取中实体关系识别研究［D］ ．合肥：中国科学技术大学， 2010．
[6]车万翔，刘挺，李生．实体关系自动抽取［ J］．中文信息学报，2005（2）：1－6．
[7]ＲOUBEＲTSA，GAIZAUSKASＲ，HEPPLEM，etal．Extracting clinicalrelationshipsfrompatientnarratives［J］．BMCBioinformat-ics，2008，9Suppl11（June）：S3． 
[8]DEMNEＲ－FUSHMANAAD，APOSTOLOVAE，ISLAMAJDＲ， etal．NLM’ssystemdescriptionforthefourthI2B2/VAchallenge［C］//Proceedingsofthe2010I2B2/VAWorkshoponChallengesinNaturalLanguageProcessingforClinicalData， 2010．
[9]DeBＲUIJNB，CHEＲＲYC，KIＲITCHENKOS，etal．Machine－ learnedsolutionsforthreestagesofclinicalinformationextraction：thestateoftheartatI2B22010［J］．JournaloftheAmericanMedicalIn-formaticsAssociation，2011，18（5）：557–562． 
[10］WANGX，CHUSEDA，ELHADADN，etal．Automatedknowl-edgeacquisitionfromclinicalnarrativereports［J］．AMIAAnnualSymposiumproceedingsAMIASymposium，2008：783–787．
[11]FＲUNZAO，INKPEND．Extractionofdisease－treatmentsemantic relationsfrombiomedicalsentences［C］//Proceedingsofthe2010WorkshoponBiomedicalNaturalLanguageProcessing，2010，（Ju-ly）：91–98．


## 电子病历在药物治疗领域的应用
### 不良药物事件（Adverse Drug Events）
不良药物事件仍然是世界各地药物治疗中发病和死亡的一个主要原因。很多药物不良反应在药物上市前临床试验中没有被发现。[8]

许多不良反应是很罕见的，只在很少的人群中发生。

不过，作为上市后监管的一个重要环节，大量药物上市后的不良事件报告机制（adverse event reporting systems, AERSs）提供了一个研究大量患者人群药物效应的机会。但是，共同使用的药物、患者人口统计学、患者病史等信息在不良事件报告系统中很难体现，这些也限制了量化特征提取分析。

两个数据库：
* 药物综合效应数据库（comprehensive database of drug effects , OFFSIDES）[9]  http://tatonettilab.org/resources/tatonetti-stm.html
* 药物相互作用引发副作用数据库（database of drug-drug interaction side effects, TWOSIDES）[9]

基于临床副作用的药物相互作用标签传播预测：
Supplements
Label Propagation Prediction of Drug-Drug Interactions Based on Clinical Side-Effects
https://astro.temple.edu/~tua87106/ddi.html
（包含多个药物相互作用和副作用数据库，可供查阅下载）

### 药物引发的肝损伤（Drug-induced liver injury, DILI）

药物引发的肝损伤一直过去50年是最频繁的药物安全相关的问题[12]。开了了一种基于深度学习架构的药物引发肝损伤模型，对475个药物的训练集训练模型，使用198个药物的验证集进行验证，准确率（accuracy）86.9%,敏感性（sensitivity）82.5%，特异性（specificity）92.9%，AUC值0.955[10]

超过700种药物发现与肝损伤相关[11]

### 药物标签的主题建模
背景：美国食品和药物管理局（FDA）批准的药品标签包含广泛的信息，从不良药物反应（ADR）到药物疗效，风险 - 收益考虑等等。但是，那用于描述这些信息的标签语言通常是含有模糊语义的自由文本描述，这对于从标签文本中检索有用的信息是一个很大的挑战以及跨药物比较分析的准确时尚。因此，这项任务在很大程度上依赖于手册由专家阅读全文，耗时耗力。方法：在这项研究中，一种具有无监督学习性质的新型文本挖掘方法称为话题建模应用于药物标签，目的是发现将相似安全问题和/或类似药物归为一类的“主题”治疗用途一起。本研究共使用794个FDA批准的药物标签。首先，三个标签每个药物标签的部分（即盒装警告，警告和注意事项，不良反应）均经过处理用于规范活动的医疗字典（MedDRA）将每个标签的自由文本转换为标准ADR条款。接下来，应用潜在狄利克雷分配（LDA）的主题建模方法生成100个主题，每一个都与基于概率分析分组在一起的一组药物相关联。最后，该疗法的功效主题建模基于已知的关于药物的治疗用途和安全性数据的信息进行评估。结果：结果表明按主题分组的药物与相同的安全问题和/或相关治疗用药有统计学意义（P <0.05）。确定的主题具有可以直接关联的独特背景特定不良事件（例如肝损伤或肾损伤）或治疗应用（例如全身使用的抗感染药）。我们还能够通过主题识别可能由特定药物引起的潜在不良事件。结论：主题建模在FDA药物标签上的成功应用证明了其潜力作为假设生成的效用意味着要推断概念的隐藏关系，例如在本研究中，药物安全和生物医学文献研究中的治疗用途。[13]


### 电子病历的疾病早期预警
Rothman指数：
尽早发现即将发生的心脏或肺部停搏是医院试图提高护理质量的重要焦点。不幸的是，目前所有的预警系统都有高误报率。大多数系统基于修改的早期预警评分（MEWS）; 5个输入中有4个是生命体征。本研究的目的是比较MEWS与Rothman指数（RI）的准确性，Rothman指数是基于利用电子病历（EMR）额外数据的超额风险函数总和的患者敏锐度分数。回顾性地计算了32,472名患者就诊的MEWS和RI评分。护理评估是RI仅使用的一类EMR输入，在死亡前24小时显示出明显差异。 24小时死亡率的接受者操作特征曲线显示出优异的RI性能，c-统计分别为0.82和0.93。在MEWS触发警报的时候，我们确定了对应相同灵敏度的RI点，发现MEWS的阳性似然比（LR +）为7.8，RI为16.9，误报降低了53％。在相当于LR +的RI点上，MEWS的敏感性为RI的49％和77％，其中24小时内死亡的患者将多捕获54％。[14]


## 参考文献
[1] Hersh, W. R. Adding value to the electronic health record through secondary use of data for quality assurance, research, andsurveillance. Am. J. Manag. Care 13, 277–278 (2007)

[2] Tatonetti, N. P., Ye, P. P., Daneshjou, R. & Altman, R. B. Data-driven prediction of drug effects and interactions. Sci. Transl. Med. 4,125ra131 (2012).

[3] Li, L. et al. Identification of type 2 diabetes subgroups through topological analysis of patient similarity. Sci. Transl. Med. 7, 311ra174(2015).

[4] Doshi-Velez, F., Ge, Y. & Kohane, I. Comorbidity clusters in autism spectrum disorders: an electronic health record time-seriesanalysis. Pediatrics 133, e54–63 (2014).

[5] Miotto, R. & Weng, C. Case-based reasoning using electronic health records efficiently identifies eligible patients for clinical trials. J. Am. Med. Inform. Assoc. 22, E141–E150 (2015).

[6] Jensen, P. B., Jensen, L. J. & Brunak, S. Mining electronic health records: towards better research applications and clinical care. Nat. Rev. Genet. 13, 395–405 (2012).

[7] Weiskopf, N. G., Hripcsak, G., Swaminathan, S. & Weng, C. Defining and measuring completeness of electronic health records for secondary use. J. Biomed. Inform. 46, 830–836 (2013).

[8] Classen DC, Pestotnik SL, Evans RS, Lloyd JF, Burke JP. Adverse drug events in hospitalized patients. Excess length of stay, extra costs, and attributable mortality. JAMA.1997; 277:301–306. [PubMed: 9002492]

[9] (Tatonetti et al., 2012) Data-driven prediction of drug effects and interactions. Sci Transl Med. PMID: 22422992 DOI: 10.1126/scitranslmed.3003377

[10] Xu Y, Dai Z, Chen F, et al. Deep Learning for Drug-Induced Liver Injury[J]. Journal of Chemical Information and Modeling, 2015, 55(10): 2085-2093.

[11] Hoofnagle, J. H.; Serrano, J.; Knoben, J. E.; Navarro, V. J. Livertox: A Website on Drug-Induced Liver Injury. Hepatology 2013, 57, 873−874
[12] Assis, D. N.; Navarro, V. J. Human Drug Hepatotoxicity: A Contemporary Clinical Perspective. Expert Opin. Drug Metab. Toxicol. 2009, 5, 463−473.
[13] Bisgin H, Liu Z, Fang H, et al. Mining FDA drug labels using an unsupervised learning technique - topic modeling[J]. BMC Bioinformatics, 2011, 12(10): 1-8.
[14] Finlay G D, Rothman M J, Smith R A, et al. Measuring the modified early warning score and the Rothman Index: Advantages of utilizing the electronic medical record in an early warning system[J]. Journal of Hospital Medicine, 2014, 9(2): 116-119.


