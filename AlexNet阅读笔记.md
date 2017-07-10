# 论文《4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf》阅读笔记 #
              作者：Alex Krizhevsky 、Ilya Sutskever 、Geoffrey E. Hinton

## 数据库介绍 ##

ImageNet拥有一千多万带标记的高分辨率图片，包含了22000中类别，本论文就是基于该数据库中的120万张图，包含50000个验证图和150000个测试图，并参加了ILSVRC-2012 的比赛，获得了比较好的结果。
该比赛比较重要的两个指标，top-1 和 top-5 错误率。
 1. **top-1**：图像实际类别不是预测的结果中分数最高的一个的错误率。
 2. **top-5**：图像实际类别不在预测结果前5中的错误率。

## 模型框架 ##

 包含了8个学习层，其中5个卷积层、3个全连接层。

 1. 非线性函数 ReLU
