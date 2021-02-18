import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import datetime

#######make a set of two combined box-plots according to April - October before 2020

#read the data from Sharston
path = 'MAHG_2.csv'
#read the data from Picc
#path = 'MAN3_copy.csv'
data = pd.read_csv(path)
data["date"] = pd.to_datetime(data['date'])
data = data.sort_values(by='date',ascending=True)
data = data.set_index('date')
# print (data.head())

#read the traffic data in Picc
#frame_traff = pd.read_csv(r'Picc.csv')
#read the traffic data in Sharston
frame_traff = pd.read_csv(r'Sharston.csv')
frame_traff = frame_traff[frame_traff['LaneDescription']=='Channel 1']
frame_traff['date'] = pd.to_datetime(frame_traff['Sdate'])
frame_traff = frame_traff.sort_values(by='date',ascending=True)
frame_traff=frame_traff.set_index('date')
#combine traffic data with air pollution data
combined_df=pd.merge(frame_traff,data, left_index=True, right_index=True)
combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]

combined_df["NO2"] = pd.to_numeric(combined_df["NO2"])
combined_df["O3"] = pd.to_numeric(combined_df["O3"])
combined_df["Volume"] = pd.to_numeric(combined_df["Volume"])
combined_df["NO2 per Volume"]=combined_df["NO2"]/combined_df["Volume"]
combined_df["log NO2 per Volume"]=np.log(combined_df["NO2"]/combined_df["Volume"])

mask_tag4 = (combined_df.index.month == 4) & (combined_df.index > '2019-12-30')
combined_df['April_2020'] = mask_tag4
mask4 = (combined_df.index.month == 4)
mask_tag5 = (combined_df.index.month == 5) & (combined_df.index > '2019-12-30')
combined_df['May_2020'] = mask_tag5
mask5 = (combined_df.index.month == 5)
mask_tag6 = (combined_df.index.month == 6) & (combined_df.index > '2019-12-30')
combined_df['June_2020'] = mask_tag6
mask6 = (combined_df.index.month == 6)
mask_tag7 = (combined_df.index.month == 7) & (combined_df.index > '2019-12-30')
combined_df['July_2020'] = mask_tag7
mask7 = (combined_df.index.month == 7)
mask_tag8 = (combined_df.index.month == 8) & (combined_df.index > '2019-12-30')
combined_df['August_2020'] = mask_tag8
mask8 = (combined_df.index.month == 8)
mask_tag9 = (combined_df.index.month == 9) & (combined_df.index > '2019-12-30')
combined_df['September_2020'] = mask_tag9
mask9 = (combined_df.index.month == 9)
mask_tag10 = (combined_df.index.month == 10) & (combined_df.index > '2019-12-30')
combined_df['October_2020'] = mask_tag10
mask10 = (combined_df.index.month == 10)
mask_tag11 = (frame_traff.index.month == 11) & (frame_traff.index > '2019-12-30')
frame_traff['November_2020'] = mask_tag11
mask11 = (frame_traff.index.month == 11)

#combined_df.to_csv('check.csv')

booleanDictionary = {True: 'TRUE', False: 'FALSE'}
f, ax = plt.subplots(4,1,figsize=(15, 15))
sns.boxplot(data=combined_df.loc[mask5],x=combined_df.loc[mask5].index.hour, y=combined_df.loc[mask5]['NO2'],hue='May_2020', ax=ax[0])
sns.boxplot(data=combined_df.loc[mask5],x=combined_df.loc[mask5].index.hour, y=combined_df.loc[mask5]['O3'],hue='May_2020',ax=ax[1])
sns.boxplot(data=combined_df.loc[mask5],x=combined_df.loc[mask5].index.hour, y=combined_df.loc[mask5]['Volume'],hue='May_2020',ax=ax[2])
sns.boxplot(data=combined_df.loc[mask5],x=combined_df.loc[mask5].index.hour, y=combined_df.loc[mask5]['log NO2 per Volume'],hue='May_2020',ax=ax[3])
ax[0].tick_params(axis='x', which='both', bottom='off',labelbottom='off')
ax[1].tick_params(axis='x', which='both', bottom='off',labelbottom='off')
ax[2].tick_params(axis='x', which='both', bottom='off',labelbottom='off')
plt.show()



f, ax = plt.subplots(4,2, figsize=(15,15))
sns.boxplot(data=combined_df.loc[mask4],x=combined_df.loc[mask4].index.hour, y=combined_df.loc[mask4]['Volume'],hue='April_2020', ax=ax[0,0])
sns.boxplot(data=combined_df.loc[mask5],x=combined_df.loc[mask5].index.hour, y=combined_df.loc[mask5]['Volume'],hue='May_2020',ax=ax[0,1])
sns.boxplot(data=combined_df.loc[mask6],x=combined_df.loc[mask6].index.hour, y=combined_df.loc[mask6]['Volume'],hue='June_2020',ax=ax[1,0])
sns.boxplot(data=combined_df.loc[mask7],x=combined_df.loc[mask7].index.hour, y=combined_df.loc[mask7]['Volume'],hue='July_2020',ax=ax[1,1])
sns.boxplot(data=combined_df.loc[mask8],x=combined_df.loc[mask8].index.hour, y=combined_df.loc[mask8]['Volume'],hue='August_2020',ax=ax[2,0])
sns.boxplot(data=combined_df.loc[mask9],x=combined_df.loc[mask9].index.hour, y=combined_df.loc[mask9]['Volume'],hue='September_2020',ax=ax[2,1])
sns.boxplot(data=combined_df.loc[mask10],x=combined_df.loc[mask10].index.hour, y=combined_df.loc[mask10]['Volume'],hue='October_2020',ax=ax[3,0])
sns.boxplot(data=frame_traff.loc[mask11],x=frame_traff.loc[mask11].index.hour, y=frame_traff.loc[mask11]['Volume'],hue='November_2020',ax=ax[3,1])

plt.show()
plt.close('all')
pdb.set_trace()

