# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:00:50 2021

@author: infiw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from xgboost import plot_importance

class Facebook_UserEngagementMetrics_Prediction:
    
    def Visualization(file_name):
        data = pd.read_csv(file_name)
        fig = plt.figure( )
        data['status_type'].value_counts().plot.pie(subplots=True)
        plt.show()
        v = np.mean(data[data['status_type']=='video'])
        p = np.mean(data[data['status_type']=='photo'])
        s = np.mean(data[data['status_type']=='status'])
        l = np.mean(data[data['status_type']=='link'])
        fig_ = plt.figure( )
        v.plot.bar(title = "Average engagements in video posts", rot = 45)
        plt.show()
        p.plot.bar(title = "Average engagements in photo posts", rot = 45)
        plt.show()
        s.plot.bar(title = "Average engagements in status posts", rot = 45)
        plt.show()
        l.plot.bar(title = "Average engagements in link posts", rot = 45)
        plt.show()
    
    def Preprocess(file_name):
        data = pd.read_csv(file_name)
        if data.isnull().values.any() == True:
            data = data.dropna(axis = 1)   
        data['status_published'] = pd.to_datetime(data['status_published']).astype('int64')
        data = data.drop(labels = 'status_id', axis = 1)
        type_class = {'video':0, 'photo':1, 'status':2,'link':3}
        data['status_type'] = data['status_type'].map(type_class)
        return data
    
    def Common_prediction(data_set, Classifer_1, Classifer_2):
        X_data = pp.scale(data_set.iloc[:, 1:])
        y_data = np.array(data_set['status_type'])
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)   
        models = [Classifer_1, Classifer_2]
        models_name = [str(Classifer_1), str(Classifer_2)]
        score_= []
        for name,model in zip(models_name, models):
            model = model   
            model.fit(X_train, y_train) 
            acc = model.score(X_test, y_test)
            score_.append(str(acc))
            print(name +' Accuracy: '+ str(acc))
            
    def XGboost_prediction(data_set, params):
        X_data = pp.scale(data_set.iloc[:, 1:])
        y_data = np.array(data_set['status_type'])
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)  
        xgb = xgboost.train(params, xgboost.DMatrix(X_train, y_train))
        y_pre = xgb.predict(xgboost.DMatrix(X_test))
        y_prob = np.argmax(y_pre, axis = 1)
        predictions = [round(value) for value in y_prob]
        xgb_acc = accuracy_score(y_test, predictions)
        print('Xgboost: ' + str(xgb_acc))
        plot_importance(xgb)
        plt.show()
        
    Visualization("Live.csv")    
    data = Preprocess("Live.csv")
    ovo = OneVsOneClassifier(LogisticRegression())
    rf = RandomForestClassifier()
    params = {'learning_rate': 0.85,
              'max_depth': 5,
              'num_boost_round': 1000,
              'objective': 'multi:softprob',
              'num_class': 4,
              'eta': 0.1
              }
    print("Show results")
    Common_prediction(data, ovo, rf)    
    XGboost_prediction(data, params)

