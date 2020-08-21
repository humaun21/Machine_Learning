#!/usr/bin/python3
# Give python version you want to use
# Import necessary libraries 
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn import metrics
from xgboost import plot_importance
from matplotlib import pyplot

import matplotlib.pylab as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 7, 3

from datetime import datetime
print('Start Time: ', str(datetime.now()))

import csv
import gc
#Read large file by chunks (e.g each time 5000 samples/rows)
def get_data(directory, file_list, chunksize, data_type_list = None, use_col_list = None):
    # type: (str, list, list, list) -> pd.DataFrame
    """
    :param directory: input directory location
    :param file_list: list of files to read
    :param data_type_list: list of data types for each file (ex. int, str, etc)
    :param use_col_list: list of columns to read from
    :return: dataframe of all files
    """

    data = None
    for i, file_name in enumerate(file_list):
        #print (file_name)
        subset = None
        for j, chunk in enumerate(
                pd.read_csv(directory + file_name, chunksize=chunksize, low_memory=False, index_col=0)):
            #print (j)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if data is None:
            data = subset.copy()
        else:
            data = pd.merge(data, subset.copy(), on="Id")
        del subset
        gc.collect()

    return data
#Load data from csv file
train = get_data('/home/abedin/bosh_analysis/important_feature/input/', ['train_numeric.csv'], 50000)
print(train.head())
y_train = train['Response']
train=train.drop(['Response'], axis =1)
columns=train.columns

# Get values
X_all=train.iloc[:, 1:train.shape[1]-1].values

# Drop train data frame and make it empty by collecting gc
del(train)
gc.collect()

def modelfit(alg, X_all, y_all, useTrainCV=True, cv_folds=3, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_all, label=y_all)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=1)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_all, y_all, eval_metric=['auc'])
    
    print(alg)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_all)
    dtrain_predprob = alg.predict_proba(X_all)[:,1]
        
    #Print model report:
    sorted_idx = np.argsort(alg.feature_importances_)[::-1]
    f_nmae = []
    score = []
    print('Important Features:')
    for index in sorted_idx:
        print([columns[index], alg.feature_importances_[index]])
#Parameters
xgb1 = XGBClassifier(learning_rate=0.05,
                     base_score=0.0056,
                     n_estimators=50,
                     max_depth=10,
                     min_child_weight=1,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='binary:logistic',
                     nthread=4,
                     scale_pos_weight=3)
# Lets go for model training                     
modelfit(xgb1, X_all, y_train)








