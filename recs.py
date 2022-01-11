import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn.ensemble import AdaBoostRegressor as Ada
from sklearn.ensemble import RandomForestRegressor as RF
from xgboost import XGBRegressor as XGB

#Extract and Pre-processing
csv_data = pd.read_csv('recs2009_public.csv')
csv_data.info()

y = (np.array(csv_data['KWH'])).astype(float)
x_data = csv_data.select_dtypes(include='number')
KWH_index = np.argwhere(np.array(x_data) == y[0])
x = np.delete(np.array(x_data), 0 & 835, axis = 1) #835 from index  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)


#prediction
models=[GB(),RF(),Ada(),XGB()]
models_str=['GradientBoost','AdaBoost','RandomForest','XGBoost']
score_=[]

for name,model in zip(models_str,models):
    print('start：'+name)
    model = model   
    model.fit(x_train,y_train)
    y_pre = model.predict(x_test)  
    acc = model.score(x_test,y_test)
    score_.append(str(acc)[:5])
    print(name +' Accuracy: '+ str(acc))
    
    plt.plot(y_test, 'b-', label = 'consumption')
    plt.plot(y_pre, 'ro', label = 'prediction')
    plt.legend()
    plt.xlabel('DOEID'); 
    plt.ylabel('Consumption');
    plt.title(name)
    plt.show()




