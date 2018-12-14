import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

#Importing data
# dataframe
df = pd.read_csv("Danish Kron.txt", sep='\t')
print(df.axes)

#homework
def replacezerowith(array, average):
    array[array == 0] = average

dfcopy = df.copy()
dfasarray = np.asarray(dfcopy.VALUE)
replacezerowith(dfasarray, np.nan)
dfcopy['VALUE'] = dfasarray
dfcopy['AVERAGE'] = dfcopy.VALUE.interpolate()
print(dfcopy)
#when we look at the new RMSE's, they provided smaller errors

size = len(dfcopy)
train = dfcopy[0:size-200]
test = dfcopy[size-200:]

df.DATE = pd.to_datetime(df.DATE,format="%Y-%m-%d")
df.index = df.DATE 
train.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
train.index = train.DATE 
test.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
test.index = test.DATE


#Naive approach
print("Naive")
dd= np.asarray(train.AVERAGE)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
rms = sqrt(mean_squared_error(test.AVERAGE, y_hat.naive))
print("RMSE: ",rms)

#Simple average approach
print("Simple Average")
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['AVERAGE'].mean()
rms = sqrt(mean_squared_error(test.AVERAGE, y_hat_avg.avg_forecast))
print("RMSE: ",rms)

#Moving average approach
print("Moving Average")
windowsize = 60
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['AVERAGE'].rolling(windowsize).mean().iloc[-1]
rms = sqrt(mean_squared_error(test.AVERAGE, y_hat_avg.moving_avg_forecast))
print("RMSE: ",rms)

# Simple Exponential Smoothing
print("Simple Exponential Smoothing")
y_hat_avg = test.copy()
alpha = 0.0
fit2 = SimpleExpSmoothing(np.asarray(train['AVERAGE'])).fit(smoothing_level=alpha,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
rms = sqrt(mean_squared_error(test.AVERAGE, y_hat_avg.SES))
print("RMSE: ",rms)

# Holt
print("Holt")
sm.tsa.seasonal_decompose(train.AVERAGE).plot()
result = sm.tsa.stattools.adfuller(train.AVERAGE)
# plt.show()
y_hat_avg = test.copy()
alpha = 0.03
fit1 = Holt(np.asarray(train['AVERAGE'])).fit(smoothing_level = alpha,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.AVERAGE, y_hat_avg.Holt_linear))
print("RMSE: ",rms)

# Holt-Winters
print("Holt-Winters")
y_hat_avg = test.copy()
seasons = 10
fit1 = ExponentialSmoothing(np.asarray(train['AVERAGE']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.AVERAGE, y_hat_avg.Holt_Winter))
print("RMSE: ",rms)
