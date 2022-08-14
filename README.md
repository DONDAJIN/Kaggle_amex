## Kaggle-amex
## Reference
このDisucussionは参考になりそう
https://www.kaggle.com/competitions/amex-default-prediction/discussion/328565

## Log 
## 2022/08/04 
・EDA.ipynb amexコンペに参入。データがでかすぎて取り込むのに苦労した。parquetファイルで圧縮されているらしい。
とりあえず今日はEDAと言いつつもただデータを取り込んだだけだった…

## 2022/08/05
・gitを少し使えるようになってすごく嬉しい。（8時間の犠牲）これからが楽しみ

## 2022/08/05
EDA.ipynbに内容を追加し、修正のみ

## 2022/08/07
* モデルの予測値というのは加重平均で扱うべきではない。https://www.kaggle.com/competitions/amex-default-prediction/discussion/329103
* 詳しくは以下もhttps://myenigma.hatenablog.com/entry/20140606/1402116869

* Lightgbmで一回目の提出をしたが、lbの真ん中ぐらいだった。各月での差の特徴量とか効きそうだけどな～

## 2022/08/09
* eda2.ipynb NaNのデータの場所に特徴がありそう。上手く0埋めしたりして情報を落とさず特徴量を作れれば精度が上がりそう。
* 集約関数でNaN=-1として計算してるから改良の余地あり。
* 欠損値の埋め方が分からない、、statementが一つしかないものは0埋めして他は回帰モデルで埋めてみるか。。
https://www.kaggle.com/code/shashankasubrahmanya/missing-data-imputation-using-regression/notebook

## 2022/08/10
* adversarial validationをやってみた。2019/4がpublic、2019/10がprivateと予想がされていて次には選んだ特徴量からモデルを作ってみようと思う。
* ノイズが[0,0.01]の大きさとしてデータが作られているため、その大きさで予測がされないように少数二桁まで丸める発想が効きそう。
* ラグ特徴量を作ってみる

## 2022/08/13
* ラグ特徴量はめちゃくちゃ効いてそう.(Validationのスコア的に) 
* LightGBMのValidationの時間がかかりすぎて事故でなかなかモデルを作れない...
* LGBM,XGBOOST,NNでアンサンブルを行いたい
* KNN,時系列での移動平均,target encodingで特徴量は作ってみたい
https://www.kaggle.com/code/heyspaceturtle/beware-the-spaceturtles

## 2022/08/14
* exp003.ipynb:最初に少数二桁に丸めてから集約や時系列の最後の特徴量、ラグ特徴量を作った。CV:0.7975 LB:0.787
* exp001.ipynb:集約や時系列の最後の特徴量、ラグ特徴量を作ってから少数二桁に丸め。　CV:0.7984 LB:0.787
* 明らかに過学習してるわ...LGBMのdartじゃ過去の決定木を上書きして学習を進めるからearly_stopping_roundsが効かないらしい。
⇒gbdtで大量に実験するしかなさそう。
