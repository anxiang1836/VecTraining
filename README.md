# Gensim的word2vec训练练习

## 1. 准备工具

```
conda install gensim
pip install jieba
```

## 2. 准备语料库

https://www.sogou.com/labs/resource/cs.php

我使用的是`特别版[王灿辉WWW08论文]数据, 647KB)`，数据类型为`.txt`，数据格式如下：

```
<doc>
<url>页面URL</url>
<docno>页面ID</docno>
<contenttitle>页面标题</contenttitle>
<content>
页面内容
</content>
</doc>
```

## 3.训练方式

采用了如下三种：

-   方式1：基于hierarchical-softmax的Skip-Gram
    
-   方式2：基于Negative-Sampling的CBOW
    
-   方式3：基于Negative-Sampling的Skip-Gram
    

其中，涉及到的训练参数为：

-   size=50
    
-   window = 5
    
-   negative = 3
    

## 4.训练速度

从时间上来看，NS-CBOW < NS-SkipGram < HS-SkipGram

| 训练方式 | 训练时间(s) |
| --- | --- |
| hs-SkipGram | 5.626199479 |
| ns-CBOW | 1.342450392 |
| ns-SkipGram | 2.375259492 |

![](http://puj4jbr34.bkt.clouddn.com/trainning-time.png)

## 5.结果比较

选择了词汇”火箭“，选用不同的模型输出了与其匹配度最高的10个词汇。

| Top-10 | HS-SkipGram | NS-CBOW | NS-SkipGram |
| --- | --- | --- | --- |
| 1 | 爵士 0.9044 | 爵士 0.9858 | 爵士 0.9223 |
| 2 | 姚明 0.8268 | 序幕 0.95733 | 火箭队 0.8505 |
| 3 | 季后赛 0.8124 | 第一战 0.9394 | 猛龙 0.8377 |
| 4 | 火箭队 0.8120 | 一号 0.9337 | 爵士队 0.8277 |
| 5 | 麦迪 0.7761 | 8475 0.9174 | 季后赛 0.8190 |
| 6 | 爵士队 0.7662 | 提防 0.9170 | NBA 0.7975 |
| 7 | 第一战 0.7551 | 犹他 0.9164 | 姚明 0.7940 |
| 8 | 姚麦 0.7429 | 季后赛 0.9141 | 开门红 0.7778 |
| 9 | NBA 0.7401 | 姚明 0.9022 | 常规赛 0.7774 |
| 10 | 统治 0.7328 | 全中 0.8985 | 麦迪 0.7704 |
