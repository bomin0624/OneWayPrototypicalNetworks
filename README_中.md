# 單向原型網絡
- 這是依據 [單向原型網絡](https://arxiv.org/abs/1906.00820) 所進行的實作，並進行了部分微調和修改。
- 數據集來自 [京都大學網頁文檔引導語料庫](https://github.com/ku-nlp/KWDLC) ，我們將數據集修改為二元分類（KWDLC-R）。
- 我們根據句子間的關係"CONTINGENCY.Cause"，將標籤設為yes(有關)，以及no(無關)。

## 特性

- 利用CL-tohoku BERT基於日語的語言模型。
- 計算話語關係句子(成對)的嵌入。
- 計算原型網絡的前向傳播。
- 提供精確度、召回率和F1分數等評估指標。
- 包括支持計算和獲取支持，以及對查詢批次的嵌入。

## 模型 
- [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)


## 需求

```
pip install -r requirements.txt
```

## 數據集

|      Data Split     |    Size    | 
|:-------------------:|:----------:|
|    Train Data       |    2087    | 
|  Validation Data    |    261     |
|     Test Data       |    262     | 


## 損失函數(Loss Function)

- 二元交叉熵(binary cross-entropy)

## 常態分佈和機率計算

在此實作中，使用常態分佈來模擬支持嵌入的平均值和查詢嵌入之間的相似性。根據這些嵌入之間的距離，計算數據點屬於該類的概率。

### 常態分佈

正態分佈，也稱為高斯分佈，是一種對稱的、鐘形的概率分佈。它由兩個參數：平均值（μ）和標準差（σ）來描述。在你的代碼中，使用固定平均值為0.0和標準差為 `self.std` 的常態分佈。

正態分佈的概率密度函數（PDF）定義為：


$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$



## 訓練
```
python oneway_protoNet.py
```

## 測試
```
python oneway_test.py
```


## 嵌入可視化

```
python visualize_proto.py
```

## 參考文獻
- A. Kruspe, One-way prototypical networks. arXiv preprint arXiv:1906.00820, 2019.
- Snell, Jake, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in neural information processing systems, 2017.
- 岸本裕大, 村脇有吾, 河原大輔, 黒橋禎夫. 日本語談話関係解析：タスク設計・談話標識の自動認識・ コーパスアノテーション, 自然言語処理, Vol.27, No.4, pp.889-931, 2020.

## 貢獻者
- Kohei Oda - [Github](https://github.com/IEHOKADO)
- Heine Chu

## 聯繫方式
如有任何疑問，請隨時發送電子郵件至 [bominchuang@jaist.ac.jp](mailto:bominchuang@jaist.ac.jp)
