# 基于 CNN, LSTM, MLP 的情感分析

## 项目结构

```
.
├── CNN_main.py
├── CNN_test.py
├── Dataset
│   ├── test.txt
│   ├── train.txt
│   ├── validation.txt
│   └── wiki_word2vec_50.bin
├── LSTM_main.py
├── LSTM_test.py
├── MLP_main.py
├── Model
│   ├── CNN.py
│   ├── LSTM.py
│   └── MLP.py
├── README.md
├── Torch
│   ├── *.pt
├── preprocess.py
├── report.md
├── report.pdf
└── utils.py
```

`./Torch` 目录保存了预处理后数据集的 tensor, 以及训练好模型的 tensor.

`./Model` 目录是三个模型的结构定义.

`./Dataset` 目录是测试/训练/验证数据集, 以及預训练好的词向量.

`*_main.py` 是训练对应模型的 py 文件. 训练后的模型会保存到 `./Torch` 目录下.

`*_test.py` 是读取训练好的 py 文件, 并输出测试集的 Loss 和 Accuracy. 注意要与训练模型的超參一致.

`preprocess.py` 是把 `./Dataset` 目录下的数据集转换成 tensor 并保存到 `./Torch` 目录下.

## 项目环境

gensim~=4.3.2

scipy~=1.12.0

torch~=2.3.0+cu118

## 代码运行

1. 执行 `python3 preprocess.py` 在 `./Torch` 中生成测试, 训练, 验证集的 tensor 文件.
2. 执行 `python3 {CNN/LSTM/MLP}_main.py` 训练模型, 并最后给出测试集的结果.
3. 执行 `python3 {CNN/LSTM/MLP}_test.py` 对训练出的模型进行测试.