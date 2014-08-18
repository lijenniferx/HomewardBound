from __future__ import division
import pandas as pd
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from MySQLPets import csv_to_mysql

data = pd.read_csv('Weather.csv')
data['Date'] = pd.to_datetime(data['DATE'],format = '%Y%m%d')
data['Month'] = map(lambda(x):x.month, data['Date'].tolist())
data['Year'] = map(lambda(x):x.year, data['Date'].tolist())
data['Day'] = map(lambda(x):x.day, data['Date'].tolist())


def replace_value_with_nan(x):
    x[np.where(np.array(x) == -9999)[0]] = np.nan
        

data.apply(lambda(x):replace_value_with_nan(x))

## convert to Farenheit
data['TMAX'] = data['TMAX']/10 * 9/5 +32
data['TMIN'] = data['TMIN']/10 *9/5 + 32

data = data[['Date','PRCP','TMAX','TMIN','Month','Year','Day']]


new_data = data.groupby('Date').apply(lambda(x):np.mean(x))
def prior_days_avg(x,days):
    x =  [x[0] for i in range(days)] + x
    values = scipy.signal.convolve(x, [1 for i in range(days)])/days
    return values[days:-(days-1)]

new_data['RAIN'] = prior_days_avg(new_data.PRCP.tolist(),10)
new_data['HEAT'] = prior_days_avg(new_data.TMAX.tolist(),10)
new_data['COLD'] = prior_days_avg(new_data.TMIN.tolist(),10)

new_data = new_data[['RAIN','HEAT','COLD', 'Month','Year','Day']]

new_data.Month = new_data.Month.apply(int)
new_data.Year = new_data.Year.apply(int)
new_data.Day = new_data.Day.apply(int)

new_data = new_data.dropna()


new_data.to_csv('PetWeather.csv',index = False)
csv_to_mysql('PetWeather.csv','PetWeather')


