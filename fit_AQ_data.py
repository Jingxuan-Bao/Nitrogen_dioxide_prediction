##########################################################################################
#    Script to fit and predict using the Facebook time series model                      #
#    Also includes ability to plot diurnal profile comparisons                           #
#                                                                                        #
#    This is free software: you can redistribute it and/or modify it under               #
#    the terms of the GNU General Public License as published by the Free Software       #
#    Foundation, either version 3 of the License, or (at your option) any later          #
#    version.                                                                            #
#                                                                                        #
#    This is distributed in the hope that it will be useful, but WITHOUT                 #
#    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS       #
#    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more              #
#    details.                                                                            #
#                                                                                        #
#    You should have received a copy of the GNU General Public License along with        #
#    this repository.  If not, see <http://www.gnu.org/licenses/>.                       #
#                                                                                        #
##########################################################################################
# 2020, author David Topping: david.topping@manchester.ac.uk

import pyreadr
import os.path
import os
import requests
import pdb
import wget
import pandas as pd
import numpy as np
import datetime
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
'''来计算RMSE'''
from sklearn.metrics import mean_squared_error, r2_score

#################################################################################
# Load an existing .csv file with air quality data for fitting. In this example
'''读取MAN3的数据'''
frame_aq = pd.read_csv('MAN3.csv')
frame_aq['datetime'] = pd.to_datetime(frame_aq['date'])
frame_aq = frame_aq.sort_values(by='datetime',ascending=True)
frame_aq=frame_aq.set_index('datetime')

#################################################################################


################################################################################
# Now load the traffic data from TfGM_Drakewell

# These files give us an insight into representiveness of traffic data
# The following data is from the closes 'Journey Time' BLU measurement point to
# the AURN site.
'''读取交通数据'''
'''交通数据只截止到15/05'''
frame_traff = pd.read_csv(r'pvr_2016-01-01_1597d_portland.csv')
# The additional option below is for a site on Portland street but away from Piccadilly Gardens
#frame_traff = pd.read_csv(r'C:\Users\Dave\Documents\Code\Developing\Traffic_analysis\TfGM_Drakewell\pvr_2016-01-01_1597d_1.csv')

# Extract data from Channel 1
'''读取channel 1的交通数据，并按时间顺序排列数据'''
frame_traff = frame_traff[frame_traff['LaneDescription']=='Channel 1']
frame_traff['datetime'] = pd.to_datetime(frame_traff['Sdate'])
'''数据按时间排序'''
frame_traff = frame_traff.sort_values(by='datetime',ascending=True)
frame_traff=frame_traff.set_index('datetime')

'''合并交通数据和空气污染数据'''
# Now merge the two dataframes, both Air Quality and Traffic, on the time index
combined_df=pd.merge(frame_traff,frame_aq, left_index=True, right_index=True)
#remove duplicate entries in the index (downloaded multiple CSV files with overlapping times)
combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]

#Now produce a box-plot for all entries in dataset.
combined_df["NO2"] = pd.to_numeric(combined_df["NO2"])
combined_df["Volume"] = pd.to_numeric(combined_df["Volume"])
combined_df["NO2 per Volume"]=combined_df["NO2"]/combined_df["Volume"]
combined_df["log NO2 per Volume"]=np.log(combined_df["NO2"]/combined_df["Volume"])
combined_df["PM2.5"][combined_df["PM2.5"]<0.0] = 0.0
combined_df["PM2.5"][combined_df["PM2.5"]>100.0] = 24.0

######################  Train a Prophet instance to the NO2 per traffic volume ###########################
train_dataset2= pd.DataFrame()
train_dataset2['ds'] = (pd.to_datetime(combined_df['Sdate']))
train_dataset2['O3']=combined_df['O3']
train_dataset2['y']=combined_df['log NO2 per Volume']
#train_dataset2['y']=combined_df['NO2 per Volume']
train_dataset2['Modelled Wind Direction']=combined_df['wd']
train_dataset2['Modelled Wind Speed']=combined_df['ws']
train_dataset2['Modelled Temperature']=combined_df['temp']
train_dataset2['Traffic Volume']=combined_df['Volume']
train_dataset2['NO2']=combined_df['NO2']
train_dataset2['Modelled PM2.5']=combined_df['PM2.5']
train_dataset2['Modelled O3']=combined_df['O3']
train_dataset2 = train_dataset2[train_dataset2.ds != 'End']
train_dataset2 = train_dataset2[train_dataset2['O3'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['y'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['Modelled Wind Direction'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['Modelled Wind Speed'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['Modelled Temperature'] != 'No data']
#train_dataset2 = train_dataset2[train_dataset2['Modelled PM2.5'] != 'No data']
train_dataset2=train_dataset2.replace([np.inf, -np.inf], np.nan)
train_dataset2.dropna(inplace=True)
pro_regressor2= Prophet()
# Add additional regressors
pro_regressor2.add_regressor('Modelled Wind Direction')
pro_regressor2.add_regressor('Modelled Wind Speed')
pro_regressor2.add_regressor('Modelled Temperature')
pro_regressor2.add_regressor('Modelled PM2.5')
pro_regressor2.add_regressor('Modelled O3')
#pro_regressor2.add_regressor('Traffic Volume')
'''划分几个时间段'''
mask_reg1b = (train_dataset2.ds < '2020-4-01')
mask_reg2b = (train_dataset2.ds >= '2020-4-01')
mask_reg3b = (train_dataset2.ds >= '2020-3-25')
mask_futureb = (train_dataset2.ds > '2019-12-01')

# Specify a train and test dataset. Train before March 2020
'''选择四月份前的数据作为训练数据'''
train_X2= train_dataset2.loc[mask_reg1b]
test_X2= train_dataset2.loc[mask_reg2b]

pro_regressor2.fit(train_X2)
forecast_data2 = pro_regressor2.predict(test_X2)
# From the forecast log(No2/Volume), use the traffic data to predict the levels of NO2
'''把NO2 per volume转化为NO2数据'''
forecast_data2['NO2 from volume']=np.exp(forecast_data2['yhat'].values)*test_X2['Traffic Volume'].values
forecast_data2['NO2 from volume upper']=np.exp(forecast_data2['yhat_upper'].values)*test_X2['Traffic Volume'].values
forecast_data2['NO2 from volume lower']=np.exp(forecast_data2['yhat_lower'].values)*test_X2['Traffic Volume'].values


#########################################################################################

###########  Train to predict NO2, as per our standard Propher use  #########################
# Now fit two prophet models, one to the just met data and one to the traffic
train_dataset= pd.DataFrame()
train_dataset['ds'] = (pd.to_datetime(combined_df['Sdate']))
train_dataset['O3']=combined_df['O3']
train_dataset['y']=combined_df['NO2']
train_dataset['Modelled Wind Direction']=combined_df['wd']
train_dataset['Modelled Wind Speed']=combined_df['ws']
train_dataset['Modelled Temperature']=combined_df['temp']
train_dataset['Traffic Volume']=combined_df['Volume']
train_dataset['NO2 per Volume']=combined_df['NO2 per Volume']
train_dataset['Modelled PM2.5']=combined_df['PM2.5']
train_dataset['Modelled O3']=combined_df['O3']
train_dataset = train_dataset[train_dataset.ds != 'End']
train_dataset = train_dataset[train_dataset['O3'] != 'No data']
train_dataset = train_dataset[train_dataset['y'] != 'No data']
train_dataset = train_dataset[train_dataset['Modelled Wind Direction'] != 'No data']
train_dataset = train_dataset[train_dataset['Modelled Wind Speed'] != 'No data']
train_dataset = train_dataset[train_dataset['Modelled Temperature'] != 'No data']
#train_dataset = train_dataset2[train_dataset['Modelled PM2.5'] != 'No data']
train_dataset=train_dataset.replace([np.inf, -np.inf], np.nan)
train_dataset.dropna(inplace=True)
mask_reg1 = (train_dataset.ds < '2020-4-01')
mask_reg2 = (train_dataset.ds >= '2020-4-01')
mask_reg3 = (train_dataset.ds >= '2020-3-25')
mask_future = (train_dataset.ds > '2019-12-01')

# Build a regressor [using a changpoint scale inferred from the Cross Validation studies]
pro_regressor= Prophet(changepoint_prior_scale=10)
#pro_regressor.add_country_holidays(country_name='UK')
pro_regressor.add_regressor('Modelled Wind Direction')
pro_regressor.add_regressor('Modelled Wind Speed')
pro_regressor.add_regressor('Modelled Temperature')
pro_regressor.add_regressor('Modelled O3')
pro_regressor.add_regressor('Modelled PM2.5')
'''选取4月前的数据作为训练数据'''
train_X= train_dataset.loc[mask_reg1]
test_X= train_dataset.loc[mask_reg2]

pro_regressor.fit(train_X)
forecast_data = pro_regressor.predict(test_X)

'''画一张测量值，NO2预测值，NO2 / traffic预测值三合一的图'''
# Plot the actual data with forecast NO2 AND NO2 from log(NO2/volume) fits
fig =pro_regressor.plot(forecast_data, uncertainty=True,figsize=(15, 5), xlabel='Date', ylabel=r'NO2 $\mu g.m^{-3}$')
plt.plot(train_dataset.loc[mask_reg2]['ds'], train_dataset.loc[mask_reg2]['y'], color='r', label='Measured')
plt.plot(forecast_data['ds'], forecast_data['yhat'], color='tab:blue', label='Forecast')
plt.plot(forecast_data2['ds'], forecast_data2['NO2 from volume'], color='g', label='Forecast using traffic data')
ax = fig.gca()
ax.set_xlim([datetime.date(2020, 4, 1), datetime.date(2020, 5, 10)])
ax.set_ylim([0, 120])
ax.set_xlabel("Date", size=14)
ax.set_ylabel(r'NO2 $\mu g.m^{-3}$', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
#plt.title('Validation data v. forecast ')
plt.legend(prop={"size":14});
plt.show()
plt.close('all')

'''
# Calculate the % deviation from the predictions based on traffic
forecast_data2['% deviation']=(forecast_data2['NO2 from volume'].values-forecast_data['yhat'].values)/(forecast_data['yhat'].values)*100.0
mask_reg4 = (forecast_data2.ds >= '2020-3-25')
'''

# Create a boxplot looking at measured, forecast with and without traffic
# To do this we are going to concatenate vertically
forecast_normal_df = forecast_data[['ds','yhat']]
forecast_normal_new_df = forecast_normal_df.copy()
forecast_normal_new_df['label'] = 'Forecast'
forecast_normal_new_df=forecast_normal_new_df.rename(columns={"yhat": "y"})
forecast_traffic_df = forecast_data2[['ds','NO2 from volume']]
forecast_traffic_new_df = forecast_traffic_df.copy()
forecast_traffic_new_df['label'] = 'Forecast using traffic data'
forecast_traffic_new_df=forecast_traffic_new_df.rename(columns={"NO2 from volume": "y"})
measured_df = train_dataset.loc[mask_reg2][['ds','y']]
measured_new_df = measured_df.copy()
measured_new_df['label'] = 'Measured'

vertical_stack = pd.concat([forecast_normal_new_df, forecast_traffic_new_df], axis=0)
vertical_stack = pd.concat([vertical_stack, measured_new_df], axis=0)

vertical_stack['ds']=pd.to_datetime(vertical_stack['ds'])
vertical_stack=vertical_stack.set_index('ds')
# Now set the index to be the datetime
f, ax = plt.subplots(1,1,figsize=(12, 5))
sns.boxplot(data=vertical_stack,x=vertical_stack.index.hour, y=vertical_stack['y'],hue='label')
ax.set_xlabel("Hour", size=14)
ax.set_ylabel(r'NO2 $\mu g.m^{-3}$', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
#plt.title('Validation data v. forecast ')
plt.legend(prop={"size":14});
plt.show()

################################Qauntify the result#####################################
mse1 = mean_squared_error(train_dataset.loc[mask_reg2]['y'],forecast_data['yhat'])
r21 = r2_score(train_dataset.loc[mask_reg2]['y'],forecast_data['yhat'])
rmse1 = np.sqrt(mse1)
print('MSE is', mse1)
print('R^2 is', r21)
print('RMSE is', rmse1)

mse2 = mean_squared_error(train_dataset.loc[mask_reg2]['y'],forecast_data2['NO2 from volume'])
r22 = r2_score(train_dataset.loc[mask_reg2]['y'],forecast_data2['NO2 from volume'])
rmse2 = np.sqrt(mse1)
print('MSE is', mse2)
print('R^2 is', r22)
print('RMSE is', rmse2)