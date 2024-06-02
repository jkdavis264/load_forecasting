
# https://data.noaa.gov/onestop/collections/details/666b9db2-e598-4d5a-9240-c4c10fc3c7a6
# https://dataminer2.pjm.com/feed/inst_load

# loads the packages to be used
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.linear_model import LinearRegression
from  sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from scipy.stats import shapiro 
import statistics
import os
from datetime import datetime
import seaborn as sb
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

#Get the current directory and changes it to the project directory
os.getcwd()
os.chdir("C:\Projects\pjm_load_forecast")
cwd = os.getcwd()

#Filters weather to past year
weather_data = pd.read_csv("data\johnglenn_weather.csv")
weather_data = weather_data[weather_data["Date"] >= "2023-05-01"]
weather_data['Date'] = pd.to_datetime(weather_data['Date'], format = "%Y-%m-%d")
weather_data.info()

#Filters the data to just PJM RTO and then gets daily values from hourly load
pjm_rto_load = pd.read_csv("data\hrl_load_prelim.csv")
pjm_rto_load['datetime_beginning_ept'] = pd.to_datetime(pjm_rto_load['datetime_beginning_ept'])

# Makes the datetime a date
pjm_rto_load['date'] = pjm_rto_load['datetime_beginning_ept'].dt.date
pjm_rto_load['load_hourly'] = pd.to_numeric(pjm_rto_load['prelim_load_avg_hourly'])
pjm_rto_load['date'] = pd.to_datetime(pjm_rto_load['date'])
pjm_rto_load.info()

#Gets the sum for the day across all regions and hours in the day
pjm_rto_load_day = pjm_rto_load[['date', 'load_hourly']].groupby('date').agg(day_load = ('load_hourly','sum')).reset_index()
pjm_rto_load_day['date'] = pd.to_datetime(pjm_rto_load_day['date'])
pjm_rto_load_day.info()

#Gets the dataset all together
pjm_data = pjm_rto_load_day.merge(weather_data, left_on='date', right_on='Date')[['date', 'day_load', 'TAVG (Degrees Fahrenheit)']]
pjm_data['month'] = pd.DatetimeIndex(pjm_data['date']).month
pjm_data['season'] = np.where(pjm_data['month'].isin([12,1,2]), "Winter", np.where(pjm_data['month'].isin([3,4,5]), "Spring", np.where(pjm_data['month'].isin([6,7,8]), "Summer","Fall")))
pjm_data.info()
del pjm_rto_load_day
del weather_data

## EDA
# Look at densities and check for NAs
pjm_data['TAVG (Degrees Fahrenheit)'].isna().sum()
sb.kdeplot(pjm_data['TAVG (Degrees Fahrenheit)'])
sb.kdeplot(data = pjm_data, x = 'TAVG (Degrees Fahrenheit)', hue="season")
sb.lineplot(data=pjm_data, x="date", y='TAVG (Degrees Fahrenheit)')

pjm_data['day_load'].isna().sum()
sb.kdeplot(pjm_data['day_load'])
sb.kdeplot(data = pjm_data, x = 'day_load', hue="season")
sb.lineplot(data=pjm_data, x="date", y='day_load')
sb.pairplot(pjm_data, diag_kind='kde')

#Look at autocorrelations
plot_acf(pjm_data['day_load'], lags=50)
plot_pacf(pjm_data['day_load'], lags=50)

#Gets the additional values
pjm_data['day_load_lag1'] = pjm_data['day_load'].shift(1)
pjm_data['day_load_lag2'] = pjm_data['day_load'].shift(2)
pjm_data['degree'] = pjm_data['TAVG (Degrees Fahrenheit)']
pjm_data['degree_cdd'] = np.where(pjm_data['TAVG (Degrees Fahrenheit)'] - 70 > 0, pjm_data['TAVG (Degrees Fahrenheit)'] - 70, 0)
pjm_data['degree_hdd'] = np.where(60 - pjm_data['TAVG (Degrees Fahrenheit)'] > 0, 60 - pjm_data['TAVG (Degrees Fahrenheit)'], 0)
pjm_data['weekday'] = pjm_data['date'].dt.dayofweek
pjm_data['weekend'] = np.where(pjm_data['weekday'].isin([5,6]), True, False)
pjm_data = pjm_data.drop(columns=['TAVG (Degrees Fahrenheit)', 'month', 'weekday'])
pjm_data.info()

#Goal to forecast the last week of each season
pjm_data_test = pjm_data.groupby('season').tail(7)
pjm_data_train = pjm_data.drop(pjm_data.groupby('season').tail(7).index)
pjm_data_train = pjm_data_train[pjm_data_train['day_load_lag2'].notna()]

#TS Model
test_x = pjm_data_test.drop(columns=['date','day_load','season'])
test_y = pjm_data_test['day_load']
x = pjm_data_train.drop(columns=['date','day_load','season'])
y = pjm_data_train['day_load']
model = LinearRegression()
model.fit(x, y)
load_hat_ts = model.predict(x)
residuals_ts = pjm_data_train['day_load'] - load_hat_ts
np.square(residuals_ts).mean()
residuals_ts_std = (residuals_ts - residuals_ts.mean())/statistics.stdev(residuals_ts)
sm.qqplot(residuals_ts_std)
shapiro(residuals_ts)
sb.lineplot(x=pjm_data["date"], y=residuals_ts_std)
plt.show()
sb.lineplot(x=load_hat_ts, y=residuals_ts_std)
plt.show()

#Model selection
x2 = sm.add_constant(x)
est = sm.OLS(y, x2.astype(float))
print(est.fit().summary())

#TS Model
x2 = x2.drop(columns=['day_load_lag2'])
est = sm.OLS(y, x2.astype(float))
print(est.fit().summary())

x2 = x2.drop(columns=['degree_cdd'])
est = sm.OLS(y, x2.astype(float))
print(est.fit().summary())


test_x2 = sm.add_constant(test_x.drop(columns=['degree_cdd','day_load_lag2']))
ts_mse_test = np.square(test_y  - est.fit().predict(test_x2)).mean() 

model = LinearRegression()
model.fit(x2, y)
load_hat_ts = model.predict(x2)
residuals_ts = pjm_data_train['day_load'] - load_hat_ts
ts_mse = np.square(residuals_ts).mean()
residuals_ts_std = (residuals_ts - residuals_ts.mean())/statistics.stdev(residuals_ts)
sm.qqplot(residuals_ts_std)
plt.show()
shapiro(residuals_ts)
sb.lineplot(x=pjm_data["date"], y=residuals_ts_std)
plt.show()
sb.lineplot(x=load_hat_ts, y=residuals_ts_std)
plt.show()

#Random forest model
model = RandomForestRegressor(n_estimators=200, max_depth=15, max_features='sqrt')
model.fit(x, y)
load_hat_rf = model.predict(x)
residuals_rf = pjm_data_train['day_load'] - load_hat_rf
rf_mse = np.square(residuals_rf).mean()
residuals_rf_test = pjm_data_test['day_load'] - model.predict(test_x)
rf_mse_test = np.square(residuals_rf_test).mean()

#Neural network
# Define Sequential model with 3 layers
scaler =  StandardScaler()
x3 = x2.drop(columns = ['const'])
x3 = scaler.fit_transform(x)
x3 = np.array(x3)
y2 = np.array([y]).transpose()
model = keras.Sequential()
model.add(keras.layers.Dense(3, activation = 'relu', input_dim=6))
model.add(keras.layers.Dense(3, activation = 'relu'))
model.add(keras.layers.Dense(1))
model.compile(loss="mean_squared_error", optimizer="sgd")
# This builds the model for the first time:
model.fit(x3, y2, epochs=10)
load_hat_nn = model.predict(x3)
residuals_nn = y2 - load_hat_nn
nn_mse = np.square(residuals_rf).mean()
residuals_nn_test = np.array([pjm_data_test['day_load']]).transpose() - model.predict(test_x)
nn_mse_test = np.square(residuals_rf_test).mean()



#Evaluate methods for forecasting load
mse = [{}]




missing_vals = pjm_data['TAVG (Degrees Fahrenheit)'].isna().sum()
print("Number of NAs in Degrees of weather :", missing_vals)
sb.kdeplot(pjm_data['TAVG (Degrees Fahrenheit)'])
plt.show()
sb.kdeplot(data = pjm_data, x = 'TAVG (Degrees Fahrenheit)', hue="season")
plt.show()
sb.lineplot(data=pjm_data.drop(columns = ['month']), x="date", y='TAVG (Degrees Fahrenheit)')













