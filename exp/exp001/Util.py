import os
from xml.sax.handler import feature_namespace_prefixes
import numpy as np
import pandas as pd
from abc import ABCMeta

class AbstractBaseBlock(ABCMeta):
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

class agg_min(AbstractBaseBlock):
    def __init__(self,column):
        self.column=column
    
    def transform(self,input_df):
        out_df=pd.DataFrame()
        out_df[self.column]=input_df.groupby('customer_ID').min()[self.column]
        return out_df.add_suffix('_min')
class agg_max(AbstractBaseBlock):
    def __init__(self,column):
        self.column=column
    def transform(self,input_df):
        out_df=pd.DataFrame()
        cid=input_df['cusomer_ID'].values
        out_df[self.column]=input_df.groupby('customer_ID').max()[self.column]
        return out_df.add_suffix('_max')

class agg_avg(AbstractBaseBlock):
    def __init__(self,column):
        self.column=column
    
    def transform(self,input_df):
        out_df=pd.DataFrame() 
        out_df[self.column]=input_df.groupby('customer_ID').mean()[self.column]
        return out_df.add_suffix('_avg')

class last_state(AbstractBaseBlock):
    def __init__(self,column):
        self.column=column
    
    def transform(self,input_df):
        out_df=pd.DataFrame()
        cid=pd.Categorical(input_df.pop('customer_ID'),ordered=True) 
        last=(cid!=np.roll(cid,-1))
        out_df=input_df.loc[last,self.column].rename(columns={f'{f}_last' for f in self.column}).set_index(np.asarray(cid[last]))
        return out_df

class num_observations(AbstractBaseBlock):#最初に実行
    def __init__(self,column):
        self.column=column
    
    def transform(self,input_df):
        out_df=pd.DataFrame()
        cid=pd.Categorical(input_df.pop('customer_ID'),ordered=True)
        out_df['num_of_observations']=input_df.groupby('customer_ID')['customer_ID'].transform('count')
        return out_df


