import os
from xml.sax.handler import feature_namespace_prefixes
import numpy as np
import pandas as pd
import datetime
import warnings
import gc
import logging
from colorama import Fore,Back,Style
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibrationDisplay
from lightgbm import LGBMClassifier,early_stopping
from lightgbm.callback import _format_eval_result


DIR='/content/drive/MyDrive/Kaggle'
EXP=os.path.join(DIR,'exp/exp001')
INPUT=os.path.join(DIR,'Input')
OUTPUT=os.path.join(DIR,'Output')
LOG=os.path.join(EXP,'Log')

os.makedirs(EXP,exist_ok=True)
for i in ['Log','Model','pred']:
    os.makedirs(os.path.join(EXP,i),exist_ok=True)

INFERENCE=True #cvだけしたいときはFalse
logger=logging.getLogger('main')
logger.setLevel(logging.DEBUG)
sc=logging.StreamHandler()
logger.addHandler(sc)
fh=logging.FileHandler(LOG)
logger.addHandler(fh)

def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback

#--------metric define------------
def amex_metric(y_true:np.array,y_pred:np.array)->float:
    
    #count of positives and negatives
    n_pos=y_true.sum()
    n_neg=y_true.shape[0]-n_pos

    indices=np.argsort(y_pred)[::-1] #降順のインデックスを取り出す
    preds,target=y_pred[indices],y_true[indices]

    #negativeには20倍の重み,positiveは1をかけて上位4％をフィルターする
    weight=20.0-target*19.0
    cum_norm_weight=(weight/weight.sum()).cumsum() #weight込みのそれぞれの値の割合をcumsumで足していく。
    four_pct_filter=cum_norm_weight<=0.04
    
    #上位4％にあるpositiveの割合
    d=target[four_pct_filter].sum()/n_pos

    lorentz=(target/n_pos).cumsum() 
    gini=((lorentz-cum_norm_weight)*weight).sum()
    
    gini_max=10*n_neg*(1-19/(n_pos+20*n_neg))

    #normilizied gini coefficient
    g=gini/gini_max

    return 0.5*(g+d)

def lgb_amex_metric(y_true,y_pred)->float:
    return ('amex',amex_metric(y_true,y_pred),'True')

#------------feature engineering-----------------------------------

features_avg = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15',
 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 
 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59',
  'D_60', 'D_61', 'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86',
   'D_91', 'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 
   'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145',
    'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 'R_26',
     'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']

features_min = ['B_2', 'B_4', 'B_5', 'B_9', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_20', 'B_28', 'B_29', 'B_33', 'B_36', 
'B_42', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70',
 'D_71', 'D_74', 'D_75', 'D_78', 'D_83', 'D_102', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 
 'D_140', 'D_141', 'D_144', 'D_145', 'P_2', 'P_3', 'R_1', 'R_27', 'S_3', 'S_5', 'S_7', 'S_9', 'S_11', 'S_12', 'S_23', 'S_25']

features_max = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 
'B_21', 'B_23', 'B_24', 'B_25', 'B_29', 'B_30', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 
'D_47', 'D_48', 'D_49', 'D_50', 'D_52', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_63', 'D_64', 'D_65', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 
'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_91', 'D_102', 'D_105', 'D_107', 'D_110', 'D_111', 'D_112', 'D_115', 'D_116', 'D_117', 'D_118', 'D_119', 
'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_138', 'D_140', 'D_141', 'D_142', 'D_144', 
'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_3', 'R_5', 'R_6', 'R_7', 'R_8', 'R_10', 'R_11', 'R_14', 'R_17', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_7', 'S_8', 
'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']

features_last = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 
'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 
'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59',
 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 'D_80', 'D_81', 'D_82',
  'D_83', 'D_86', 'D_91', 'D_96', 'D_105', 'D_106', 'D_112', 'D_114', 'D_119', 'D_120', 'D_121', 'D_122', 'D_124', 'D_125', 'D_126', 'D_127', 'D_130', 
  'D_131', 'D_132', 'D_133', 'D_134', 'D_138', 'D_140', 'D_141', 'D_142', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7',
   'R_8', 'R_9', 'R_10', 'R_11', 'R_12', 'R_13', 'R_14', 'R_15', 'R_19', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 
   'S_13', 'S_16', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']

for i in ['test','train'] if INFERENCE else['train']:
    df=pd.read_parquet(os.path.join(INPUT,f'{i}.parquet'))
    cid=pd.Categorical(df.pop('customer_ID'),ordered=True) #'customer_IDをカテゴリ化
    last=(cid!=np.roll(cid,-1)) #np.rollでインデックスを-1ずらし、各customerの最後のstatementを取り出す
    if 'target' in df.columns:
        df.drop(columns=['target'],inplace=True)
    gc.collect()
    print('Read',i)
    
    df_avg=(df.groupby(cid).mean()[features_avg].rename(columns={f:f'{f}_avg' for f in features_avg}))
    gc.collect()
    print('Computed avg',i)

    df_min=(df.groupby(cid).min()[features_min].rename(columns={f:f'{f}_min' for f in features_min}))
    gc.collect()
    print('Computed min',i)

    df_max=(df.groupby(cid).max()[features_max].rename(columns={f:f'{f}_max' for f in features_max}))
    gc.collect()
    print('Computed max',i)

    df=(df.loc[last,features_last].rename(columns={f:f'{f}_last' for f in features_last}).set_index(np.asarray(cid[last]))) 
    #カテゴリであるcid[last](='customer_ID')をindexにする
    gc.collect()
    
    #統合する
    df=pd.concat([df,df_min,df_max,df_avg],axis=1)
    if i=='train':train=df
    else:test=df
    print(f'{i} shape: {df.shape}')
    del df,df_avg,df_min,df_max,cid,last

target=pd.read_csv(os.path.join(INPUT,'train_labels.csv'))['target'].values
print(f'target shape: {target.shape}')

#-------------------------------------------------------------------------------------

#--------------------Cross Validation-------------------------------------------------
ONLY_FIRST_FOLD=False

features=[f for f in train.columns if f!='customer_ID' and f!='target']

def my_booster(random_state=1,n_estimators=1200):
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,reg_lambda=50,
        min_child_samples=2400,
        num_leaves=95,
        colsample_bytree=0.19,
        max_bins=511,
        random_state=random_state
    )
print(f'{len(features)} features')

score_list=[]
y_pred_list=[]
kf=StratifiedKFold(n_splits=5)
for fold,(idx_tr,idx_va) in enumerate(kf.split(train,target)):
    X_tr,X_va,y_tr,y_va,model=None,None,None,None,None
    start_time=datetime.datetime.now()
    X_tr=train.iloc[idx_tr][features]
    X_va=train.iloc[idx_va][features]  #.ilocの方がRAMの消費が小さい
    y_tr=target[idx_tr]
    y_va=target[idx_va]

    model=my_booster()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=UserWarning)
        model.fit(X_tr,y_tr,
        eval_set=[X_va,y_va],
        eval_metric=[lgb_amex_metric],
        callbacks=[])
    X_tr,y_tr=None,None
    y_va_pred=model.predict(X_va,raw_score=True)
    score=amex_metric(y_va,y_va_pred)
    n_trees=model.best_iteration_
    if n_trees is None:n_trees=model.n_estimators
    print(f'{Fore.GREEN}{Style.BRIGHT}Fold {fold} | {str(datetime.datetime.now()-start_time)[-12:-7]} |'
            f'{n_trees:5} trees |'
            f' Score={score:.5f}{Style.RESET_ALL}')
    score_list.append(score)

    if INFERENCE:
        y_pred_list.append(model.predict.proba(test[features],raw_score=True))

    if ONLY_FIRST_FOLD:break #一回のFoldのスコアだけが欲しいとき

print(f'{Fore.GREEN}{Style.BRIGHT} OOF SCORE: {np.mean(score_list):.5f}{Style.RESET_ALL}')








