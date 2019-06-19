# Anomaly Detection in Videos by using Convolutional LSTM
Convolutional LSTM[1]を用いて動画異常検知を行おうという話(未達) \
異常検知としては既に参考文献[2]で行われているが，評価基準がいまいちぱっとせずよくわからない．

## Convolutional LSTM
名前の通り，LSTMの全結合層を畳み込み層に変更したもの．単純に画像をRNNあるいはLSTMにつっこむより画像の位置情報が失われにくい．参考文献[1]では天気の予測に使用している．

## 異常検知
ConvLSTMを用いて数フレームを予測する正常モデルを構築，異常データが入力されたときとの数値誤差を利用して異常検知．

## やってみたいこと
Imaging Time Series[3]を用いて時系列を画像化し，分類あるいは本ケースのような異常検知ができないか．\
ただ，時系列を画像化することに具体的なメリットが見出せないためいまいち踏み切れない．ラベル付けもめんどくさそう．

## データセット
- [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

## 参考文献
[1] S. Xingjian, Z. Chen, H. Wang, D. Yeung, W. Wong, and W. Woo. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Neural Information Processing Systems (NIPS), 2015. \
[2] J. R. Medel and A. Savakis. Anomaly detection in video using predictive convolutional long short-term memory networks. arXiv preprint arXiv:1612.00390, 2016. \
[3] Zhiguang Wang and Tim Oates. Imaging time-series to improve classification and imputation. Proceedings of the 24th International Join Conference on Artificial Intelligence (IJCAI), 2015.
