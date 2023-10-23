### A survey of the vision transformers and their CNN‑transformer based variants

​		[Review of ViT](https://arxiv.org/abs/2305.09880)

***

**0.abstract**

ViT又可能替代CNN在计算机视觉领域的许多应用。

对比CNN，Transformer可以关注到一张图片上的全局关系（CNN只能关注到卷积核覆盖的小部分之间的关系）。

而目前的ViT则更多是CNN和Attention Mechanism（Attention Mechanism是Transformer的核心）的混合，使得ViT可以同时关注到local和global的特征。

这篇文章就是介绍目前常见的ViT架构，特别是那些混合ViT架构。

***

**1.introduction**

首先介绍了计算机视觉领域的现状，并肯定了纯CNN模型为计算机视觉领域带来了革新。（作者罗列了一些CNN在CV三大领域：图片分类、物体检测和语义分割的成功应用的论文）

CNN的问题：CNN只关注local-level的空间相关性，这会导致如果要学习的特征更大、更复杂，那CNN的表现就会下降。

ViT的发展历程：从2017年[Transformer](https://arxiv.org/abs/1706.03762)的提出。到2018年[自注意力机制被首次应用于CV领域](https://arxiv.org/abs/1802.05751)，但是仅限于local层面。最后到2020年提出[ViT](https://arxiv.org/abs/2010.11929)模型，使得模型可以从全局学习图片的特征，ViT在多个数据集上表现优异，至此CV领域开始更多关注Transformer的应用。

ViT对比CNN：CNN有归纳偏置（This  inductive  bias  includes  characteristics  like  translation and  scale  invariance  due  to  the  shared  weights  across  different  spatial  locations.），主要包括locality（桌子旁边一般有椅子）和translation equivariance（一个物体不管怎么移动都能学到相同的特征），这相当于一种先验知识（prior knowledge），是图像的特点，这使得CNN可以在较小的数据集上学到图片的特征，但这也导致了CNN处理全局特特征的不完美。同时，缺乏这些归纳偏置，也导致了ViT对于dataset的规模有较高的要求，且Transformer自身对于算力也有较高的要求。

对比ViT和CNN的优势和缺陷，提出[HVT](https://arxiv.org/abs/2206.10589)：Hybrid Vision Transformers，通过卷积层提取local的特征，再通过ViT的自注意力机制提取全局的特征。

作者罗列了一些近期的介绍新ViT架构和ViT应用的论文。

这篇文章对ViT和HVT模型从架构上做了一个分类（分别是6类ViT和7类HVT），并介绍了他们在CV各个方面的应用，因为我们做的是图片分类领域的工作，因此我们也只关注ViT和HVT在图片分类领域的工作。

***

**2.Fundamentals Concepts in ViTs**

Transformer -> ViT用了三年时间，自注意力机制用于视觉任务的难点如下：

在处理nlp问题中，一个句子中的每个token，也就是一个个单词，在embedding之后进入自注意力机制中，token之间两两做互动，求出一个自注意力的图，然后经过一个MLP进行输出。其实很容易想到，将一张2D的图片的每个像素flatten后变为一个1D的序列，每个像素作为token输入自注意力机制中，这样transformer就可以用于视觉任务，但难点就在于此，因为每个token两两之间要做运算，这就导致计算复杂度非常大，一个512个词的句子只有512个token，但一张224*224的图，就有50176个token，两两之间做运算，计算量太大；而且过大的参数量容易过拟合（可以类比到卷积神经网络，如果将一张2D的图片的每个像素flatten后变为一个1D的序列输入一个全连接层，那也会导致模型的参数量过大）。

后续提出的一些解决方案：

比如只在局部的16*16窗口做自注意力（局部的思想和原来的CNN没有本质上的区别）、分别在x轴和y轴上做自注意力

***

首先介绍一下[ViT](https://arxiv.org/abs/2010.11929)的基本概念。

**ViT结构**

***

**embedded**

1. 一张3通道224×224的图片
2. 拆成196个16×16×3的patch
3. 线性投射层：全连接层，将patch转为token，将2D的图转为一个1D的token序列；每个token维度是768，共196个token
4. cls token：借鉴BERT加一个cls token，所以一共1+196=197个token，每个token维度是768
5. 为每个token加上位置编码，用的是sum而不是concat，因此不改变token的维度（positional embedding 见 D4）

**LayerNorm**

**Multi-Head Self-Attention**

base版本的ViT有12个head，因此每个k，q，v对应的大小是197×64（768 / 12 = 64）

**Residual Connection**

**LayerNorm**

**MLP**

进行一个维度放大，一般放大四倍，就是197×3072

**Residual Connection**

**output**

***

**class token**

全局表示：Class token 附加到这些图像块的向量序列的开头，经过 Transformer 的多个自注意力层后，它被用来积累全局的图像信息。这是因为在 Transformer 的操作中，每个 token（包括 class token）都有机会与其他所有 token 交互，从而捕获整个图像的上下文信息。

分类目的：在经过 Transformer 层处理后，只有 class token 的输出被用于分类任务。更具体地说，class token 的输出被送入一个前馈神经网络（通常是一个简单的线性层），以产生最终的分类预测。

***

然后介绍了HVT的核心概念

1.在Transformer的图像处理中的patching和tokenization阶段，卷积可以捕捉图像的局部特征。

2.[CvT](https://arxiv.org/abs/2103.15808)（Convolutional Vision Transformer）中，使用一个基于卷积的proj在image patches中学习空间上的和low-level的信息，然后将卷积输出线性变化为一个序列，进入Transformer结构中。CNN中有一种增加通道数，使图片尺寸变小的卷积层，称为CNN的空间降采样，而CvT使用分层布局，减少token的数量但增加token的宽度也是类似的行为（令牌池），可以减少计算量。

3.[CeiT](https://arxiv.org/abs/2103.11816)（Convolution-enhanced Image Transformers）中利用卷积操作扩展low-level的特征，通过的是image-to-token的结构。两者的主要区别在于如何结合卷积和Transformer结构。CvT使用卷积来处理低级特征，并通过层次化的Transformer结构进行长范围的信息处理。而CeiT通过混合注意机制在每一层中都结合了卷积和自注意力。

4.[CCT](https://arxiv.org/abs/2104.05704)（Compact Convolutional Transformer）中集成了卷积-池化块来进行tokenization，在小数据集上的表现较好。

5.[LocalViT](https://arxiv.org/abs/2104.05707)：将depth-wise convolution引入forward中。

6.[LeViT](https://arxiv.org/abs/2104.01136)：高速推理。

7.[CoAtNet](https://arxiv.org/abs/2106.04803)：

***

**3.Architectural level modifications in ViTs**

table3显示HVT在知名数据集上的准确率，我们做的事图像分类方面的，因此关注ImageNet数据集上的准确率，准确率最高的使CvT和[MaxViT](https://arxiv.org/abs/2204.01697)，都属于作者分类的Hierarchical integration分类。因此主要了解这两个。

table2显示的则是ViT的准确率，在ImageNet上准却屡最高的是[CaiT](https://arxiv.org/abs/2103.17239)和[TinyViT](https://arxiv.org/abs/2207.10666)。

其中的其中，TinyViT（21M）的参数规模和之前使用的EfficientNetB4（19M）类似，属于可训练的范畴。

所以我觉得重要介绍一下TinyViT。

***

**4.Applications of ViTs and HVTs**

1.图像识别

2.图像生成

3.图像分割

4.图像修复

5.特征提取

6.医疗影像分析

7.目标检测

8.位置预测