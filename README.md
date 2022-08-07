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
このdiscussion
https://www.kaggle.com/competitions/amex-default-prediction/discussion/329103
model.predictでraw_score=Trueにすることで対数オッズを算出できる。例えば二つのモデルがあって99％の予測と50％の予測のモデルの時に二つの加重平均をした時に75％になるのは感覚的におかしい。その代わりに対数オッズの加重平均を行うと安定した予測値がでる。詳しくは以下https://myenigma.hatenablog.com/entry/20140606/1402116869
