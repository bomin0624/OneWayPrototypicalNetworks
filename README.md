# One-Way Prototypical Networks
- An Implementation of [One-Way Prototypical Networks](https://arxiv.org/abs/1906.00820) with some modifications.
- The dataset is from [Kyoto University Web Document Leads Corpus](https://github.com/ku-nlp/KWDLC) and we modified the dataset to binary classification (KWDLC-R).
- We set the "CONTINGENCY.Cause" relation sentence pairs as label yes and other discourse relation to no.

## Features

- Utilizes the CL-tohoku BERT-based Japanese language model.
- Calculates embeddings for discourse relation sentences pairs.
- Computes the Prototypical Network's forward pass.
- Provides evaluation metrics such as precision, recall, and F1-score.
- Includes support for calculating and retrieving embeddings for support and query batches.

## Model 
- [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)


## Requirements

```
pip install -r requirements.txt
```

## Dataset

|      Data Split     |    Size    | 
|:-------------------:|:----------:|
|    Train Data       |    2087    | 
|  Validation Data    |    261     |
|     Test Data       |    262     | 


## Loss Function

- binary cross-entropy

## Normal Distribution and Probability Calculation

In this implementation, a normal distribution is used to model the similarity between the mean support embedding and query embeddings. The probability of a data point belonging to the positive class is computed based on the distance between these embeddings.

### Normal Distribution

A normal distribution, also known as a Gaussian distribution, is a probability distribution that is symmetric and bell-shaped. It is characterized by two parameters: the mean (μ) and the standard deviation (σ). In your code, a normal distribution with a fixed mean of 0.0 and a standard deviation of `self.std` is used.

The probability density function (PDF) of the normal distribution is defined as:


$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$



## Train
```
python oneway_protoNet.py
```

## Test
```
python oneway_test.py
```


## Embedding Visualization

```
python visualize_proto.py
```

## Reference
- A. Kruspe, One-way prototypical networks. arXiv preprint arXiv:1906.00820, 2019.
- Snell, Jake, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in neural information processing systems, 2017.
- 岸本裕大, 村脇有吾, 河原大輔, 黒橋禎夫. 日本語談話関係解析：タスク設計・談話標識の自動認識・ コーパスアノテーション, 自然言語処理, Vol.27, No.4, pp.889-931, 2020.

## Contributors
- Kohei Oda - [Github](https://github.com/IEHOKADO)

## Contact
For inquiries, please don't hesitate to email [bominchuang@jaist.ac.jp](mailto:bominchuang@jaist.ac.jp)
