# 论文资源

|    论文题目    |   关键词＋概述   | 下载路径 | 说明（创新点） |
| ----------- | ------------------- | -------- | -------|
|Exploiting Similarities among Languages for Machine Translation| 词向量，双语翻译baseline|https://arxiv.org/pdf/1309.4168.pdf|1. 使用简单线性矩阵乘法学习语言A到语言B的映射 2. 构造评测集以及构造对照组的方法。如使用编辑距离，Word Co-occurrence构造的对照组|
| Word Translation Without Parallel Data | 多语言的Embedding，对抗学习 | https://arxiv.org/pdf/1710.04087.pdf | https://github.com/facebookresearch/MUSE ，用对抗学习语言A到语言B的映射 |
|Transfer Learning for Deep Sentiment Analysis |带情感信息的Embedding，迁移学习 | http://www.aclweb.org/anthology/P18-1235 |误差函数的设计，加了正则，该正则使得学到网络对已知情感标签的词效果好；可借鉴评测方式|
|Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation | Normalized Word Embedding以及Orthogonal Transform可提高Bilingual分类性能 | http://www.aclweb.org/anthology/N15-1104 |提高模型训练效率以及稳定性的方法（normalize+正则），容易工程实现|
|Training Neural Word Embeddings For Transfer Learning And Translation|迁移学习的Embedding|http://scholar.sun.ac.za/handle/10019.1/98758||
|Wiktionary-Based Word Embeddings|wiki字典，多语言全局embedding|http://www.mt-archive.info/15/MTS-2015-DeMelo.pdf|基于词与词之间的关系（使用加权内积）建模|
|Adversarial Network Embedding|对抗网络Embedding|https://arxiv.org/pdf/1711.07838.pdf||
|Interpretable Adversarial Perturbation in Input Embedding Space for Text|对抗扰动生成方式，此对抗样本|https://www.ijcai.org/proceedings/2018/0601.pdf|对抗扰动生成方式增加限制，另其仅在有实际意义的word的方向变动，而非随机任意方向变动（后续或者可以加入更强限制，比如lm模型）|
|Enriching Word Vectors with Subword Information|char n-gram，形态学词向量|https://research.fb.com/wp-content/uploads/2017/06/tacl.pdf|使用char级别的 n-gram 特征建模，得到表征词形态的词向量|
|Linguistic Regularities in Sparse and Explicit Word Representations||http://www.aclweb.org/anthology/W14-1618|nothing special|
|Advances in Pre-Training Distributed Word Representations|词向量训练优化手段|https://arxiv.org/pdf/1712.09405|词频为Zipf分布，需要提高高频词的discard probability；类似attention的加权context embedding；可以用mutual information criterion从数量爆炸的n-gram中选择少部分信息量大的|
|Learning to Compose Words into Sentences with Reinforcement Learning|tree-structured representations,增强学习|https://arxiv.org/pdf/1611.09100||
|Recent Trends in Deep Learning Based Natural Language Processing|多个nlp任务 state of art 方法|http://2www.sentic.net/deep-learning-for-nlp-review.pdf , http://www.neo-classical-physics.info/uploads/3/0/6/5/3065888/noether_-_invariant_variational_problems.pdf|多个nlp任务最新进展，参考价值很大|
|Invariant Variation Problems|经典不变量分析|https://arxiv.org/pdf/physics/0503066|最经典论文,可以参考https://en.wikipedia.org/wiki/Noether%27s_theorem|
|Lexicon infused phrase embeddings for named entity resolution|词典，Phrase Embeddings|https://arxiv.org/pdf/1404.5367.pdf|改装skip-gram，除了预测上下文的context外，还预测辞典中与改词关联的context|
|Improved Word and Symbol Embedding for Part-of-Speech Tagging|词向量的使用技巧|https://denero.org/content/pubs/snl17_altieri_tagging.pdf|byte-pair encoding|
|Unsupervised POS Induction with Word Embeddings|HMM、多元高斯分布|http://www.aclweb.org/anthology/N15-1144|skip-gram减小window size更利于获取语法信息；假设某个tag对应的词向量符合多元高斯分布|
|Improving the Accuracy of Pre-trained Word Embeddings for Sentiment Analysis|通过简单concat的方法改进词向量|https://arxiv.org/ftp/arxiv/papers/1711/1711.08609.pdf|直接简单concat word2vec、glove、pos2vec、leicon2vec，缺点是需要引入外部有监督数据|
|Part-Of-Speech Tag Embedding for Modeling Sentences and Documents |POS Embedding|https://escholarship.org/uc/item/0vk28220|nothing special|
|Diagnosing and Enhancing VAE Models|ICLR 2019接收论文|https://openreview.net/pdf?id=B1e0X3C9tQ||
