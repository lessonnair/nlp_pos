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
|Recent Trends in Deep Learning Based Natural Language Processing|多个nlp任务 state of art 方法|http://2www.sentic.net/deep-learning-for-nlp-review.pdf|多个nlp任务最新进展，参考价值很大|
|Invariant Variation Problems|经典不变量分析|https://arxiv.org/pdf/physics/0503066 , http://www.neo-classical-physics.info/uploads/3/0/6/5/3065888/noether_-_invariant_variational_problems.pdf|最经典论文,可以参考https://en.wikipedia.org/wiki/Noether%27s_theorem|
|Lexicon infused phrase embeddings for named entity resolution|词典，Phrase Embeddings|https://arxiv.org/pdf/1404.5367.pdf|改装skip-gram，除了预测上下文的context外，还预测辞典中与改词关联的context|
|Improved Word and Symbol Embedding for Part-of-Speech Tagging|词向量的使用技巧|https://denero.org/content/pubs/snl17_altieri_tagging.pdf|byte-pair encoding|
|Unsupervised POS Induction with Word Embeddings|HMM、多元高斯分布|http://www.aclweb.org/anthology/N15-1144|skip-gram减小window size更利于获取语法信息；假设某个tag对应的词向量符合多元高斯分布|
|Improving the Accuracy of Pre-trained Word Embeddings for Sentiment Analysis|通过简单concat的方法改进词向量|https://arxiv.org/ftp/arxiv/papers/1711/1711.08609.pdf|直接简单concat word2vec、glove、pos2vec、leicon2vec，缺点是需要引入外部有监督数据|
|Part-Of-Speech Tag Embedding for Modeling Sentences and Documents |POS Embedding|https://escholarship.org/uc/item/0vk28220|nothing special|
|Diagnosing and Enhancing VAE Models|ICLR 2019接收论文|https://openreview.net/pdf?id=B1e0X3C9tQ ||
|使用生成式对抗网络进行远距离监督关系抽取||https://mp.weixin.qq.com/s/mEwJs3ayo9iSg0S2uEfWdQ ， https://arxiv.org/pdf/1805.09929|用对抗网络进行样本去噪，generator和discriminator目标相反，generator尽可能预测样本干净度，预测出来的标签取反放入discriminator训练，训练直到discriminator性能下降最大为止（discriminator初始时和generator的目标一样）。|
|Phrase-Based & Neural Unsupervised Machine Translation|非监督机器翻译|https://arxiv.org/pdf/1804.07755.pdf||
|Unsupervised Part-of-Speech Taggingwith Bilingual Graph-Based Projections|非监督 POS Tagging|https://aclanthology.info/pdf/P/P11/P11-1061.pdf||
|Distinguishing Antonyms and Synonyms in a Pattern-based Neural Network|有监督训练word pair关系|https://arxiv.org/pdf/1701.02962.pdf|借助 parse tree 特征有监督训练反义word pair，基本假设是在同一句话里，反义词同时出现的概率大于同义词同时出现的概率|
|Integrating Distributional Lexical Contrast into Word Embeddings for Antonym–Synonym Distinction|训练能区分同义词的embedding|https://arxiv.org/pdf/1605.07766.pdf|两种方法，方法一是在普通的词向量的目标函数中加上同义词和反义词的正则；方法二是给词向量的各个特征re－weight，能区分反义词的特征加大权重|
|Evaluating semantic relations in neural word embeddings with biomedical and general domain knowledge bases||https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6069806/|评测三种词向量在semantic task的效果|

# 2018 NIPS Best Paper
|    论文题目    |   关键词＋概述   | 下载路径 | 说明（创新点） |
| ----------- | ------------------- | -------- | -------|
|Non-delusional Q-learning and Value-iteration||http://120.52.51.17/www.cs.toronto.edu/~cebly/Papers/nondelusionalQ_nips18.pdf||
|Optimal Algorithms for Non-Smooth Distributed Optimization in Networks||https://papers.nips.cc/paper/7539-optimal-algorithms-for-non-smooth-distributed-optimization-in-networks.pdf||
|Nearly Tight Sample Complexity Bounds for Learning Mixtures of Gaussians via Sample Compression Schemes||https://papers.nips.cc/paper/7601-nearly-tight-sample-complexity-bounds-for-learning-mixtures-of-gaussians-via-sample-compression-schemes.pdf||
|Neural Ordinary Differential Equations||https://arxiv.org/pdf/1806.07366||

