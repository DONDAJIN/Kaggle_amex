import pandas as pd
import numpy as np
import itertools
#各特徴量のIV(Information ValueやWeight of Events)を計算できる
def iv_woe(data,target,bins=20,show_woe=False):
    #Empty DataFrame
    newDF,woeDF=pd.DataFrame(),pd.DataFrame()

    #Extract Columns Names
    cols=data.columns

    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        print(ivars)

        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>3):
            #dtype:(boolean,integer,float,complex ) and unique varibale>3
            binned_x=pd.qcut(data[ivars],bins,duplicates='drop') #ほぼ均等にbinningする。
            d0=pd.DataFrame({'x':binned_x,'y':data[target]})
        
        else:
            #カテゴリの場合はそれ自体がbinningされているとみなせる
            d0=pd.DataFrame({'x':data[ivars],'y':data[target]})
        
        d0=d0.astype({'x':str})
        d=d0.groupby('x',as_index=False,dropna=False).agg({'y':['count','sum']})
        d.columns=['Cutoff','N','Events']
        d["% of Events"]=np.maximum(d['Events'],0.5)/d['Events'].sum()
        d['Non-Events']=d['N']-d['Events']
        d['% of Non-Events']=np.maximum(d['Non-Events'],0.5)/d['Non-Events'].sum()
        d['WoE']=np.log(d['% of Non-Events']/d['% of Events'])
        d['IV']=d['WoE']*(d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0,columns='Variable',value=ivars)
        print('Infromation value of'+ivars+'is'+str(round(d['IV'].sum(),6)))
        temp=pd.DataFrame({'Variable':[ivars],'IV':[d['IV'].sum()]},columns=['Variable','IV'])
        newDF=pd.concat([newDF,temp],axis=0)
        woeDF=pd.concat([woeDF,d],axis=0)

        #Show Woe Table
        if show_woe==True:
            print(d)
    return newDF,woeDF

#permutation feature engを行う時
high=['P_10','D_21','D_43'] 
all_pairs=[]
for i in range(len(high)-1):
    all_pairs.extend(list(itertools.product([high[i]],high[i+1:])))
len(all_pairs)#組み合わせの数

#作れる特徴量の組み合わせでIVを計算していく
data=pd.DataFrame()#train_df
all_features=pd.DataFrame()
for pair in all_pairs:
    agg_df=pd.DataFrame()

    #新しい特徴量
    agg_df[f'{pair[0]}_t_{pair[1]}']=data[pair[0]]*data[pair[1]]
    agg_df[f'{pair[0]}_d_{pair[1]}']=data[pair[0]]/data[pair[1]] #0除算したらnp.infになる
    agg_df[f'{pair[0]}_p_{pair[1]}']=data[pair[0]]+data[pair[1]]
    agg_df[f'{pair[0]}_m_{pair[1]}']=data[pair[0]]-data[pair[1]]

    #時系列とかの場合は各customerとかでまとめて
    agg_df=agg_df.groupby('customer_ID').agg(['last', 'first', 'mean', 'std', 'max', 'min'])
    agg_df.columns=['_'.join(x) for x in agg_df.columns]
    agg_df=agg_df.fillna(0)
    agg_df.replace([np.inf,-np.inf],0,inplace=True)
    agg_df['target']=data['target']

    #IVを計算
    a,b=iv_woe(agg_df,'target')

    #good featuresを選択
    good_ones=a.loc[a['IV']>2.5]['Variable'].values #->変数名のndarray
    all_features[good_ones]=agg_df['good_ones']

    print(f'current shape: {all_features.shape}')
