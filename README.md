# TwoLayerNet
使用python的numpy构建两层神经网络分类器

## MINIST数据集
获取地址：http://yann.lecun.com/exdb/mnist/

下载4个文件（包括训练集和测试集的图像和标签），将其解压至./minist文件夹下

## 代码文件
### 1.训练：model1.py
模型主要部分，包括激活函数，反向传播，loss以及梯度的计算，学习率下降策略，L2正则化，优化器SGD更新权重

在当前路径下运行 python model.py，将保存初始设定参数下的分类器twolayermodel.npz以及可视化训练和测试的loss曲线，测试的accuracy曲线，每层的网络参数

### 2.参数查找：para_search.py
查找参数和范围如下

学习率：learning_rate_list = [0.001, 0.005, 0.01]

隐藏层大小：hidden_dim_list = [100, 500, 1000]

正则化强度：regularization_list = [0, 0.0001, 0.001]


在当前路径下运行 python para_search.py

查找范围内每组参数对应的accuracy写入para_search.txt，并在最终输出最优参数

当前参数范围内的最优模型保存至bestmodel.npz

### 3.测试：test.py
导入para_search.py得到的最优分类器bestmodel.npz

在MINIST测试数据集上测试分类精度

在当前路径下运行 python test.py，可输出分类器精度

## 训练模型
训练好的模型twolayermodel.npz和bestmodel.npz上传至百度网盘

链接: https://pan.baidu.com/s/183z2_eZ0OkR7WN5_0z67Fg?pwd=qmfi 提取码: qmfi
