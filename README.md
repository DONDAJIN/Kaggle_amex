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
* exp003.ipynb: 最初に少数二桁に丸めてから集約や時系列の最後の特徴量、ラグ特徴量を作った。CV:0.7975 LB:0.787
* exp001.ipynb: 集約や時系列の最後の特徴量、ラグ特徴量を作ってから少数二桁に丸め。　CV:0.7984 LB:0.787
* 明らかに過学習してるわ...LGBMのdartじゃ過去の決定木を上書きして学習を進めるからearly_stopping_roundsが効かないらしい。
⇒gbdtで大量に実験するしかなさそう。
* exp004.ipynb: target encodingで他はexp001.ipynbと一緒。(他はLabel encodingを用いた)

## 2022/08/17
* 過学習しないようにパラメータチューニングでlambda項を大きくした。
* exp015,exp011,exp013,exp010が異なるseedでのlgbmモデル。今まで過学習してたっぽかったのはもしかしたらtestデータとtrainデータの列の型が少し異なっていたからかもしれない...それを直してからはCVとLBの値はそこまで離れなくなった。それでもう一度dart試してみることにする。(exp012,exp016)
* exp015: catboost

## 2022/08/18
* やっとLBが0.799を超えた。privateの予測を上げるためにadvarsarial validationで特定した特徴量を使わないモデルを作ってそれをアンサンブルに組み込もうと思う。
* Target encodingが効いているのかは余り分からない。効いたらそれでアンサンブルするのもあり。LBよりもTrust CVって言葉もあるくらいだし慎重に。

## 2022/08/21
* {exp017: catboost, exp018: 'B_29'を使わずにfeしたLGBM,exp019: TargetEncodingしたLGBM}
* WoEとIVで特徴量の重要度を計算し、それの高いものの組み合わせで四則演算を行い、特徴量を作った。それでまたLGBMを作ってみる。

## 2022/08/23
* {exp020: WoEとIVで作った特徴量を加えてLGBM, exp021:exp20の異なるseed}
* 特徴量を減らさずに増やしたせいでRAMが足りなくなった。feature importanceで特徴量を削るのが必要。（決定木は不要な特徴量を自動で判断するけれど、分かりやすさやtrainの速さの観点的に）

## 2022/08/24
* {exp020: これまでに作ったモデルの内、LBで0.799を超えたもので適当に重みを付けてアンサンブル。

## 2022/08/25
* 結果的に688位/4935だった...選択しなかったsubの中には銀圏に食い込んでたものもあってそれで一層残念。
* solution見て勉強する。

## 2022/08/26
* 2nd  place solution: データを見たときにhistory(各顧客)の長さによってtargetの平均が異なる。->stratified group kfold を使う。大体13の長さが多かったが有効か

* 時系列データはやはりラグ特徴量が重要。最後のdata vs (5個前まで)で作ってもいいくらい。
* 'S_2'からmonthや曜日などの抽出も
* 集約特徴量やPermutationで特徴量を作る際にはやはりPCAは重要。
* 各customerのstatementの数が多いほど予測が上手くいっているからモデルを使ってstatementの補完を行い、それによってtrainを増やすことができる!!solutionではRNNを使い、一つのGRU層といくつかのFC層を用いている。特徴量は対数変換とfillna(0)をしている。しかもこれはtargetのデータが必要じゃないからtrainとtestの両方を使って予測することが出来る!!!Maxが13のstatementだからそれでtrainしてそれより少ないstatementのデータに対して予測する感じ。
* NNを使う時はやはり対数変換するなどして分布をいい感じにするべき
* 今度はNNも使いたいな

