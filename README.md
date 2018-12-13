# 论文资源

|    论文题目    |   关键词＋概述   | 下载路径 | 说明（创新点） |
| ----------- | ------------------- | -------- | -------|
|Exploiting Similarities among Languages for Machine Translation| 词向量，双语翻译baseline|https://arxiv.org/pdf/1309.4168.pdf|1. 使用简单线性矩阵乘法学习语言A到语言B的映射 2. 构造评测集以及构造对照组的方法。如使用编辑距离，Word Co-occurrence构造的对照组|
| Word Translation Without Parallel Data | 多语言的Embedding，对抗学习 | https://arxiv.org/pdf/1710.04087.pdf | https://github.com/facebookresearch/MUSE ，用对抗学习语言A到语言B的映射 |
|Transfer Learning for Deep Sentiment Analysis |带情感信息的Embedding，迁移学习 | http://www.aclweb.org/anthology/P18-1235 |误差函数的设计，加了正则，该正则使得学到网络对已知情感标签的词效果好|
|Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation | Normalized Word Embedding以及Orthogonal Transform可提高Bilingual分类性能 | http://www.aclweb.org/anthology/N15-1104 |提高模型训练效率以及稳定性的方法（normalize+正则），容易工程实现|
|Training Neural Word Embeddings For Transfer Learning And Translation|迁移学习的Embedding|http://scholar.sun.ac.za/handle/10019.1/98758||
|Wiktionary-Based Word Embeddings|wiki字典，多语言全局embedding|http://www.mt-archive.info/15/MTS-2015-DeMelo.pdf|基于词与词之间的关系（使用加权内积）建模|
|Cross-Lingual Propagation for Deep Sentiment Analysis||http://iiis.tsinghua.edu.cn/~weblt/papers/sentiment-propagation.pdf||
