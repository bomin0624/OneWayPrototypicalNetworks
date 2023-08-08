# One-Way Prototypical Networks
- An Implementation of [One-Way Prototypical Networks](https://arxiv.org/abs/1906.00820) with some modifications.
- The dataset is from [Kyoto University Web Document Leads Corpus](https://github.com/ku-nlp/KWDLC) and we modified the dataset to binary classification (KWDLC-R).
- We set the "CONTINGENCY.Cause" relation sentence pairs as label yes and other discourse relation to no.

## Dataset

|      Data Split     |    Size    | 
|:-------------------:|:----------:|
|    Train Data       |    2087    | 
|  Validation Data    |    261     |
|     Test Data       |    262     | 


## Performance on KWDLC-R

|    Model               |  Precision  |    Recall    |   F1-Score   |
|:----------------------:|:-----------:|:------------:|:------------:|
|       Rule-Based        |      0.73       |       0.34      |     0.47      |
|         BERT            |      0.79       |      0.88        |       0.83       |
| One-Way Prototypical Networks|      0.95      |       0.94       |        0.94      |


## Reference
- A. Kruspe, One-way prototypical networks. arXiv preprint arXiv:1906.00820, 2019.
- Snell, Jake, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in neural information processing systems, 2017.
- 岸本裕大, 村脇有吾, 河原大輔, 黒橋禎夫. 日本語談話関係解析：タスク設計・談話標識の自動認識・ コーパスアノテーション, 自然言語処理, Vol.27, No.4, pp.889-931, 2020.

## Contact
For inquiries, please don't hesitate to email [bominchuang@jaist.ac.jp](mailto:bominchuang@jaist.ac.jp)


