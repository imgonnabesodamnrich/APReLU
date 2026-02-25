# 论文笔记：融合自适应激活函数的深层残差网络(ResNet-APReLU) —— 面向复杂工况的机械故障诊断

> **An Adaptive Deep Learning Framework for Mechanical Fault Diagnosis**
>
> **Paper Title:** *Deep Residual Networks With Adaptively Parametric Rectifier Linear Units for Fault Diagnosis*
> 
> **Journal:** *IEEE Transactions on Industrial Electronics (TIE)*, Vol. 68, No. 3, 2021
>
> **DOI:** [10.1109/TIE.2020.2972458](https://doi.org/10.1109/TIE.2020.2972458)

[![DOI:10.1109/TIE.2020.2972458](https://img.shields.io/badge/DOI-10.1109/TIE.2020.2972458-blue.svg)](https://doi.org/10.1109/TIE.2020.2972458)
[![Status: Published](https://img.shields.io/badge/Status-Published-green.svg)]()
[![Topic: Deep Learning](https://img.shields.io/badge/Topic-Deep%20Learning-orange.svg)]()
[![Field: Fault Diagnosis](https://img.shields.io/badge/Field-Fault%20Diagnosis-blueviolet.svg)]()

---

## 目录
1. [引言与背景](#1-引言与背景)
2. [变工况下的特征学习挑战](#2-变工况下的特征学习挑战)
3. [核心创新：APReLU 激活函数](#3-核心创新aprelu-激活函数)
    * 3.1 [APReLU 的设计动机](#31-aprelu-的设计动机)
    * 3.2 [子网络结构与参数自适应](#32-子网络结构与参数自适应)
4. [ResNet-APReLU 整体架构](#4-resnet-aprelu-整体架构)
5. [实验分析与验证](#5-实验分析与验证)
    * 5.1 [行星齿轮箱实验设计](#51-行星齿轮箱实验设计)
    * 5.2 [对比实验结果](#52-对比实验结果)
    * 5.3 [特征可视化分析](#53-特征可视化分析)
6. [结论](#6-结论)

---

## 1. 引言与背景
在现代工业体系中，旋转机械（如齿轮箱、轴承）的健康监测对保障生产安全具有决定性作用。随着深度学习技术的发展，基于数据驱动的故障诊断方法已逐渐取代传统的手工特征提取。然而，实际工业环境中的机械设备往往面临复杂的工况变化，如转速的频繁波动和负载的不断切换。这些因素导致振动信号的统计特性表现出极强的非平稳性。论文提出了一种融合自适应参数化线性修正单元（APReLU）的深层残差网络模型，旨在提升模型在复杂变工况场景下的特征判别精度。

## 2. 变工况下的特征学习挑战
在变工况条件下，机械故障诊断面临两个核心挑战：
*   **同类不相似性**：同一故障类型在不同转速或负载下，其振动信号在频域或时域的分布可能存在巨大差异。
*   **异类相似性**：不同类型的故障（如齿轮磨损与轴承点蚀）在特定工况下可能表现出高度相似的振动特征。

传统的深度学习模型（如 ResNet, CNN）采用固定的非线性激活函数（如 ReLU），其对所有输入信号应用统一的非线性变换。这种“一刀切”的处理方式难以在特征空间中有效分离高度混淆的信号，限制了模型的判别边界灵活性。

<div align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/08ae33d8-6522-432d-aaf1-64a3608ac809" />
  <p><em>图1 对比示意图 (a) 传统固定变换 (b)论文提出的自适应变换 </em></p>
</div>

## 3. 核心创新：APReLU 激活函数

### 3.1 APReLU 的设计动机
为了增强模型对不同输入信号的适应能力，论文设计了自适应参数化线性修正单元（APReLU）。与传统的激活函数不同，APReLU 的负半轴斜率不再是预设的常数，也不是全局共享的学习参数，而是根据当前输入样本动态生成的变量。

### 3.2 子网络结构与参数自适应
APReLU 通过集成一个轻量级的子网络来实现参数的自适应调整。具体过程如下：
1.  **全局信息提取**：输入特征图首先通过全局平均池化（GAP）操作，提取各通道的全局空间信息。
2.  **动态斜率预测**：子网络（包含全连接层与 Sigmoid 映射）根据提取的信息，为每一个输入通道计算对应的斜率系数 $\alpha$。
3.  **非线性变换**：模型根据生成的动态斜率对当前样本进行非线性映射，实现了“一物一码”的特征变换效果。

## 4. ResNet-APReLU 整体架构
模型以深层残差网络（ResNet）为基干。ResNet 的恒等连接机制有效解决了深层神经网络中的梯度消失问题。论文将 APReLU 激活函数嵌入到每一个残差块（ResBlock）中。这种结合使得网络不仅具备深层的语义抽象能力，还能在每一层特征传递中，针对变工况样本进行动态的增益调整和特征矫正。

<div align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/35ac8e48-bdac-4841-99d2-2f60aad2976b" />
  <p><em>图2 (a) APReLU结构图 (b) 改进的ResBlock (c) 整体网络架构 </em></p>
</div>

## 5. 实验分析与验证

### 5.1 行星齿轮箱实验设计
论文利用行星齿轮箱动力学模拟实验台进行数据采集。实验设置了 8 种不同的健康状态（包括正常、齿面磨损、齿根裂纹等），并在 3 种转速与 3 种负载组合下进行测试。同时，为了评估模型的稳健性，在原始信号中加入了不同强度的加性高斯白噪声。

### 5.2 对比实验结果
实验将 ResNet-APReLU 与传统的 ConvNet 以及采用 Sigmoid, Tanh, ReLU, LReLU, PReLU 等激活函数的 ResNet 进行了横向对比：
*   **准确率提升**：在各种信噪比（5dB, 3dB, 1dB）环境下，ResNet-APReLU 的测试准确率均保持最高。
*   **泛化性能**：相比于采用固定斜率的模型，自适应模型在面对未见工况时表现出更强的鲁棒性。

### 5.3 特征可视化分析
通过 t-SNE 算法对模型最后一层特征进行降维可视化。结果表明，ResNet-APReLU 能够显著拉开异类故障之间的特征距离，并压缩同类故障在不同工况下的特征分布空间，形成更为紧凑且清晰的聚类结果。

<div align="center">
  <img width="90%" src="https://github.com/user-attachments/assets/b49a9f74-ebbc-466c-a70c-4c34675bb292" />
  <p><em>图3 不同ResNet变体与论文方法的 t-SNE 可视化对比 </em></p>
</div>

## 6. 结论
论文提出的 ResNet-APReLU 模型通过在残差结构中引入自适应激活机制，有效解决了复杂变工况下振动信号特征判别难的问题。该方法在行星齿轮箱故障诊断任务中表现优异，反映了动态调整非线性变换参数在处理非平稳工业信号中的优势，为工业大数据的智能分析提供了一种更具灵活性的神经网络构建思路。

## 文献来源：

**标题：** Deep Residual Networks With Adaptively Parametric Rectifier Linear Units for Fault Diagnosis

**期刊：** IEEE Transactions on Industrial Electronics

**DOI：** [10.1109/TIE.2020.2972458](https://doi.org/10.1109/TIE.2020.2972458)
