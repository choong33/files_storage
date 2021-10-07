# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:23:26 2021

@author: Wai
"""
# https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/custom_dataset.py
# https://neptune.ai/blog/keras-metrics
# https://www.kaggle.com/niyamatalmass/lightfm-hybrid-recommendation-system

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from collections import Counter
import scipy.sparse as sparse
from sklearn.ensemble import RandomForestClassifier
import os
import logging
#filepath = 'C:/Users/I008328/Desktop/AIA/Next Recommended Actions/'
filepath = "C:/Users/Wai\OneDrive/Desktop/insurance/"
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
# Load data
################################################################################
class MyDataset(Dataset):
    """
    A dataset of random colored graphs.
    The task is to classify each graph with the color which occurs the most in
    its nodes.
    The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
    nodes connected with probability `p`.
    """

    def __init__(self):
        pass
        
    def read_and_clean(self,df):
        df = df[df['INS_AGE'].notna()]
        df = df[df['INS_AGE'] > 1]
        mean = round(df["INS_KID_CT"].mean())
        df["INS_KID_CT"] = np.where(df["INS_KID_CT"] >5, mean,df['INS_KID_CT'])
        df['INS_L_AGT_SEGM'] = np.where(df['INS_L_AGT_SEGM'] == 'NLP',df['INS_L_AGT_SEGM'].mode()[0],df['INS_L_AGT_SEGM'])
        df_flags = df[['FLG_AGE_21','FLG_MARRIED','FLG_LIFE_STAGE','FLG_CHILD','FLG_INCOME_CHANGE','FLG_BIRTHDAY', 
                             'FLG_CLM_TRIGGER','FLG_CLM_FAM_TRIGGER','FLG_CALL_TRIGGER','FLG_MATURING','FLG_MATURED','FLG_SUM' ,'TARGET_REPUR_IND']]
        
        return df
        
    def fill_missing_values(self,df):
        for col in df.select_dtypes(include= ["int","float"]).columns:
            val = df[col].mean()
            df[col].fillna(val, inplace=True)
            return df
        
    def remove_outliers(df, column_list):
        for col in column_list:
            avg = df[col].mean()
            std = df[col].std()
            low = avg - 2 * std
            high = avg + 2 * std
            df = df[df[col].between(low, high, inclusive=True)]
        return df
    
    def create_features(self,df):
        perc =[.10,.20,.30 ,.40,.50 ,.60,.70, .80,.90,1]
        #df.INS_RECENCY.describe(percentiles = perc)
        cut_labels = ['p10r','p20_r','p30_r','p40_r','p50_r','p60_r','p70_r','p80_r','p90_r','p100_r']
        cut_bins = [-1,2,6,14,23,37,53,78,121,197,df['INS_RECENCY'].max()]
        df['Recency_Cat'] = pd.cut(df['INS_RECENCY'],bins=cut_bins,labels=cut_labels)
        
        cut_labels = ['p25_ap','p50_ap','p75_ap','p1_ap']
        cut_bins = [-1,1800,3433,6600,df['INS_IF_AP'].max()]
        df['AP_Cat'] = pd.cut(df['INS_IF_AP'],bins=cut_bins,labels=cut_labels)
        
        cut_labels = ['p1_pp','p2_pp','p3_pp','p5_pp','p10_pp','p15_pp','p20_pp','>p20_pp']
        cut_bins = [-1,1,2,3,5,10,15,20,df['INS_PP_POL'].max()]
        df['PP_Cat'] = pd.cut(df['INS_PP_POL'],bins=cut_bins,labels=cut_labels)
        
        df = pd.get_dummies(data=df,columns=['PP_Cat','AP_Cat','INS_INCOME_SEGM','INS_LIFE_STAGE','INS_L_AGT_SEGM'])
        
        features = ['INS_ID','INS_RECENCY','INS_AGE','INS_BILL_AMT','INS_PP_POL','INS_INT_MYAIA_TRAN_3M','INS_KID_CT','INS_NON_ACUTE',
                    'INS_SMOKER_IND','INS_INT_CALL_3M','INS_ETI_RPU','INS_POL_APL_PH','INS_LATE_PAYMENT_12M',
                    'INS_INCOME_SEGM_G6 Mass Affluent','AP_Cat_p50_ap','INS_INCOME_SEGM_G3 Mass','AP_Cat_p75_ap',
                    'AP_Cat_p1_ap','AP_Cat_p25_ap','INS_LIFE_STAGE_Mature Single','INS_INCOME_SEGM_G14 ENHW/HNW',
                    'INS_LIFE_STAGE_Mature Couple','INS_INCOME_SEGM_G11 Mass Affluent','INS_INCOME_SEGM_G7 Mass Affluent',
                    'INS_PAID_AMT_ACUTE_6','INS_INCOME_SEGM_G2 Mass','INS_INCOME_SEGM_G13 Mass Affluent',
                    'INS_INT_EMAIL_3M','INS_LIFE_STAGE_Established Family','INS_CLM_MAJOR','INS_CS_BAD_RISK',
                    'INS_INCOME_SEGM_G9 Mass Affluent','INS_INCOME_SEGM_G5 Mass Affluent','INS_PAID_AMT_BAD_RISK',
                    'INS_ACUTE_CT_6','INS_LIFE_STAGE_Young Couple','INS_LIFE_STAGE_Nest Builder','INS_LIFE_STAGE_Young Single',
                    'INS_INCOME_SEGM_G8 Mass Affluent','INS_LIFE_STAGE_Golden Ager','INS_INCOME_SEGM_G4 Mass','INS_BAD_RISK',
                    'INS_PAID_AMT_ACCIDENT_6','INS_INCOME_SEGM_G12 Mass Affluent','INS_INCOME_SEGM_G10 Mass Affluent',
                    'INS_ACCIDENT_CT_6','INS_PAID_AMT_BAD_RISK_24','INS_LAPSE_12M','INS_LATE_PAYMENT_3M','INS_LIFE_STAGE_Minors ',
                    'INS_CS_BAD_RISK_CT_24','INS_BAD_RISK_CT_24','INS_SUR_POL_12M','INS_POL_LOAN_IND','VIT_ACT_3M']
        
        def above_21(x):
            y = x-21
            if y > 0:
                y = y+100
            else:
                y = abs(y)
            return y

        def marital_age(x,y,z):
            i = abs(x-35)
            if y == 'Mature Single' or y == 'Young Single':
                j = i
            else:
                j = x *2 # 0
    
            if y == 'Minors ':
                j = x + 100
            
            return j
        
        df['Above_21'] = df['INS_AGE'].apply(lambda x: above_21(x))
        df['Not_Married_Before'] = np.vectorize(marital_age) \
                                          (df['INS_AGE'],df['INS_LIFE_STAGE'], \
                                           df['Above_21']) 
        
        df = pd.concat([df[features],df['Above_21'],df['Not_Married_Before']],axis=1)
    
        return df
    
    def make_interaction(self,df_flags):
        lst = ['FLG_AGE_21','FLG_MARRIED','FLG_LIFE_STAGE','FLG_CHILD','FLG_INCOME_CHANGE','FLG_BIRTHDAY','FLG_CLM_TRIGGER',
               'FLG_CLM_FAM_TRIGGER','FLG_CALL_TRIGGER','FLG_MATURING','FLG_MATURED']
        lst_rate = [0.0283,0.083,0.0447,0.0479,0.0653,0.0224,0.0261,0.0423,0.0373,0.0281,0.0355]
        
        for k,v in zip(lst,lst_rate):
    #df_train[k] = df_train[k].apply(lambda x: v if x == 1 else 0)
            df_flags[k] = df_flags[k].apply(lambda x: v if x > 0 else 0)
            
        for i in lst:
    #df_train[i] = df_train.apply(lambda x: x[i]**(x['FLG_SUM']+1) if (x['TARGET_REPUR_IND'] == 0 and x[i]>0) else x[i],axis=1)
            df_flags[i] = df_flags.apply(lambda x: x[i]*((1/x['FLG_SUM'])+1) if (x['TARGET_REPUR_IND'] == 1 and x[i]>0) else x[i],axis=1)
    #df_train[i] = df_train.apply(lambda x: -x[i] if (x['TARGET_REPUR_IND'] == 0 and x[i]>0) else x[i],axis=1)
      
        interactions = df_flags[lst]
        
        assert interactions.shape[1] == len(lst), "..."
        
        logging.info('Completed interactions matrix')
       
        return interactions
            
        
    def make_sparse(self,interactions,df):
        flags = interactions.shape[1]
        interaction_f = sparse.coo_matrix(interactions)
        user_f  = sparse.coo_matrix(df.iloc[:,1:])
        flag_f  = sparse.identity(flags)
        
        return (interaction_f,user_f,flag_f)
        
    def train(self,df):
        
        
        return
        
        
        
    def evaluate(self,df):
        
        return
        

logging.info('Reading and Preparing Data')

logging.info('Training Model')


















