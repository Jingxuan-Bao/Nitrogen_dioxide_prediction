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
frame_aq = pd.read_csv('MAN3_copy.csv')
#frame_aq = pd.read_csv('MAHG_2.csv')
#frame_aq.to_csv('frame1.csv')
#frame_aq['datetime'] = pd.to_datetime(frame_aq['date'],format='%d/%m/%Y %H:%M')
frame_aq['date'] = pd.to_datetime(frame_aq['date'])
frame_aq = frame_aq.sort_values(by='date',ascending=True)
#frame_aq.to_csv('frame_aq3.csv')
frame_aq=frame_aq.set_index('date')
#frame_aq.to_csv('frame2.csv')

#################################################################################


################################################################################
# Now load the traffic data from TfGM_Drakewell

# These files give us an insight into representiveness of traffic data
# The following data is from the closes 'Journey Time' BLU measurement point to
# the AURN site.
'''读取交通数据'''
'''交通数据只截止到15/05'''
frame_traff = pd.read_csv(r'Picc.csv')
#frame_traff = pd.read_csv(r'pvr_2016-01-01_1597d_portland.csv')
#frame_traff = pd.read_csv(r'Sharston.csv')
# The additional option below is for a site on Portland street but away from Piccadilly Gardens
#frame_traff = pd.read_csv(r'C:\Users\Dave\Documents\Code\Developing\Traffic_analysis\TfGM_Drakewell\pvr_2016-01-01_1597d_1.csv')

# Extract data from Channel 1
'''读取channel 1的交通数据，并按时间顺序排列数据'''
frame_traff = frame_traff[frame_traff['LaneDescription']=='Channel 1']
frame_traff['datetime'] = pd.to_datetime(frame_traff['Sdate'])
frame_traff['datetime'] = frame_traff['Sdate']
frame_traff['datetime'] = pd.to_datetime(frame_traff['datetime'],format='%d/%m/%Y %H:%M')
'''数据按时间排序'''
frame_traff = frame_traff.sort_values(by='datetime',ascending=True)
frame_traff = frame_traff.set_index('datetime')
#frame_traff = frame_traff['2016-3-05' : '2020-5-15']
frame_traff = frame_traff['2016-3-05' : '2019-3-27']
#frame_traff.to_csv('frame1.csv')
'''
frame_traff2 = frame_traff2[frame_traff2['LaneDescription']=='Channel 1']
frame_traff2['datetime'] = pd.to_datetime(frame_traff2['Sdate'])
#数据按时间排序
frame_traff2 = frame_traff2.sort_values(by='datetime',ascending=True)
frame_traff2 = frame_traff2.set_index('datetime')
frame_traff2 = frame_traff2['2016-3-05' : '2020-5-15']

frame_traff2.to_csv('frame_traff2.csv')
frame_traff.to_csv('frame_traff.csv')

r22 = r2_score(frame_traff['Volume'],frame_traff2['Volume'])
print ('########################: ', r22)
'''

# 合并交通数据和空气污染数据
# Now merge the two dataframes, both Air Quality and Traffic, on the time index
combined_df=pd.merge(frame_traff,frame_aq, left_index=True, right_index=True)
#combined_df.to_csv('combine1.csv')
#remove duplicate entries in the index (downloaded multiple CSV files with overlapping times)
combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]
#combined_df.to_csv('combine2.csv')
#Now produce a box-plot for all entries in dataset.
combined_df["NO2"] = pd.to_numeric(combined_df["NO2"])
combined_df["Volume"] = pd.to_numeric(combined_df["Volume"])
combined_df["NO2 per Volume"]=combined_df["NO2"]/combined_df["Volume"]
combined_df["log NO2 per Volume"]=np.log(combined_df["NO2"]/combined_df["Volume"])
combined_df["PM2.5"][combined_df["PM2.5"]<0.0] = 0.0
combined_df["PM2.5"][combined_df["PM2.5"]>100.0] = 24.0
combined_df.to_csv('combine.csv')
######################  Train a Prophet instance to the NO2 per traffic volume ###########################
train_dataset2= pd.DataFrame()
train_dataset2['ds'] = (pd.to_datetime(combined_df['Sdate'],format='%d/%m/%Y %H:%M'))
train_dataset2['O3']=combined_df['O3']
train_dataset2['y']=combined_df['NO2']
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
m = Prophet()
m.add_country_holidays(country_name='UK')
m.add_regressor('Modelled Wind Direction')
m.add_regressor('Modelled Wind Speed')
m.add_regressor('Modelled Temperature')
m.add_regressor('Modelled PM2.5')
m.add_regressor('Modelled O3')
mask_reg1b = (train_dataset2.ds < '2019-3-23')
mask_reg2b = (train_dataset2.ds >= '2019-3-23')
train_X = train_dataset2.loc[mask_reg1b]
test_X = train_dataset2.loc[mask_reg2b]
train_X.to_csv('train.csv')

#future = pd.DataFrame()
#future = test_X['ds']

m.fit(train_X)
future = m.make_future_dataframe(periods=96, freq='H')
#future.reset_index(drop=True)
future.to_csv('future.csv')
print(future.tail())

#forecast_data = m.predict(future)
forecast_data = m.predict(test_X)

fig =m.plot(forecast_data, uncertainty=True,figsize=(15, 5), xlabel='Date', ylabel=r'NO2 $\mu g.m^{-3}$')
plt.plot(test_X['ds'], test_X['y'], color='r', label='Measured')
plt.plot(forecast_data['ds'], forecast_data['yhat'], color='tab:blue', label='Forecast')
ax = fig.gca()
ax.set_xlim([datetime.date(2019, 3, 22), datetime.date(2019, 3, 29)])
#ax.set_ylim([0, 120])
ax.set_ylim([0, 100])
ax.set_xlabel("Date", size=14)
ax.set_ylabel(r'NO2 $\mu g.m^{-3}$', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
#plt.title('Validation data v. forecast ')
plt.legend(prop={"size":14});
plt.show()
plt.close('all')







