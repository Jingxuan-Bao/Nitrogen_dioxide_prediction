#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:22:50 2020

@author: baojingxuan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import datetime
from sklearn.model_selection import RandomizedSearchCV


##############################The process to read data###############################

#read the data from Sharston
#path = 'MAHG_2.csv'
#read the data from Picadically
path = 'MAN3_copy.csv'
data = pd.read_csv(path)
data["date"] = pd.to_datetime(data['date'])
data = data.sort_values(by='date',ascending=True)
data = data.set_index('date')
# print (data.head())
data["weekday"] = data.index.weekday
data["hour"] = data.index.hour
data["month"] = data.index.month

#read the traffic data
frame_traff = pd.read_csv(r'pvr_2016-01-01_1597d_portland.csv')
frame_traff = frame_traff[frame_traff['LaneDescription']=='Channel 1']
frame_traff['date'] = pd.to_datetime(frame_traff['Sdate'])
frame_traff = frame_traff.sort_values(by='date',ascending=True)
frame_traff=frame_traff.set_index('date')
#combine traffic data with air pollution data
combined_df=pd.merge(frame_traff,data, left_index=True, right_index=True)
combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]
# print (data.head())

#You can change the list to get different group of features
#list = ['O3','NO2','temp','ws']
#list = ['O3','NO2','temp','ws','wd','weekday','hour','month']
#list = ['O3','NO2','temp','ws','wd','weekday','hour','month','Volume']
#list = ['NO2','PM2.5','temp','ws','wd','weekday','hour','month']
list = ['PM2.5','NO2','temp','ws','wd','weekday','hour','month','Volume']

df= combined_df.loc[:,list]
#df = data.loc[:,list]

df.to_csv('look.csv')
#cut the train and test datasets
df_train= df.truncate(before = '2020-4-01')
df_test = df['2020-4-01 00:00:00':'2020-5-14 23:00:00']
#df_test.to_csv('check_testdata.csv')
#print(df_train.head())
#print(df_test.head())

x_train = df_train.drop(['NO2'],axis=1)
x_test = df_test.drop(['NO2'],axis=1)
y_train = df_train.loc[:,['NO2']]
y_test = df_test.loc[:,['NO2']]

'''
##########################The process to select parameters#################################

criterion=['mse','mae']
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'criterion':criterion,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

clf= RandomForestRegressor()
clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                              n_iter = 10,  
                              cv = 3, verbose=2, random_state=42, n_jobs=1)

clf_random.fit(x_train, y_train)
print (clf_random.best_params_)
'''


##############################The process of training model###########################

######the parameters for picadically with & without traffic volume input & sharston
rf=RandomForestRegressor(criterion='mae',bootstrap=True,
                         max_features='auto', max_depth=70,min_samples_split=2, 
                         n_estimators=600,min_samples_leaf=2)


rf.fit(x_train,y_train.values.ravel())
#y_train_pred=rf.predict(x_train)
y_test_pred=rf.predict(x_test)

#y_test.to_csv('predict.csv')
#print(y_test.head())
#print(y_test.shape)

###########################Find the importances of different features#################

features = x_train.columns
x = features
f_im = rf.feature_importances_
y = f_im
plt.bar(x,y)
plt.title('Feature Importances',color='black',size=12)
plt.show()

###########################Make the plot of Measured vs Prediction#####################
#add predict value to y_test dataframe
y_test['predict'] = y_test_pred
#make plot
fig = plt.figure(figsize=(15,5))
plt.plot(y_test.index,y_test['NO2'],label='Measured',color='red')
plt.plot(y_test.index,y_test['predict'],label='Predict_RF',color='tab:blue')
ax=fig.gca()
#ax.set_xlim([datetime.date(2020, 4, 1), datetime.date(2020, 5, 10)])
#ax.set_ylim([0, 80])
ax.set_ylim([0, 120])
ax.set_xlabel("Date", size=14)
ax.set_ylabel(r'NO2 $\mu g.m^{-3}$', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.show()

#############################Quantify the prediction result##########################
mse = mean_squared_error(y_test['NO2'],y_test['predict'])
r2 = r2_score(y_test['NO2'],y_test['predict'])
rmse = np.sqrt(mse)
print('MSE is', mse)
print('R^2 is', r2)
print('RMSE is', rmse)


#############################Make the box plot of Measured vs Prediction##############
#make a dataframe to give label to masured data and prediction data
y_test['ds']=y_test.index
#y_test.to_csv('check.csv')
measured_normal_df =pd.DataFrame()
measured_normal_df = y_test[['ds','NO2']]
measured_normal_new_df = measured_normal_df.copy()
measured_normal_new_df['label'] = 'Measured'
foreast_normal_df =pd.DataFrame()
foreast_normal_df = y_test[['ds','predict']]
foreast_normal_new_df = foreast_normal_df.copy()
foreast_normal_new_df['label']='Predict'
foreast_normal_new_df=foreast_normal_new_df.rename(columns={"predict": "NO2"})
vertical_stack = pd.concat([measured_normal_new_df,foreast_normal_new_df] , axis = 0)
vertical_stack = vertical_stack.set_index('ds')
#vertical_stack.to_csv('check.csv')

#make box plot
f, ax = plt.subplots(1,1,figsize=(12, 5))
sns.boxplot(data=vertical_stack,x=vertical_stack.index.hour, y=vertical_stack['NO2'],hue='label')
ax.set_xlabel("Hour", size=14)
ax.set_ylabel(r'NO2 $\mu g.m^{-3}$', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
#plt.title('Validation data v. forecast ')
plt.legend(prop={"size":14});
plt.show()
