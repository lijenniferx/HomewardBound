import pandas as pd
import numpy as np

def get_pop(dataframe):
    
    def get_expiration_date(x):
        if len(pd.date_range(x.ArrivalDate, periods = x.LengthofStay, freq = 'D')) == 0:
            return x.ArrivalDate
        else:        
            return pd.date_range(x.ArrivalDate, periods = x.LengthofStay, freq = 'D')[-1]

    
    
    dataframe['ExpirationDate'] =  dataframe.apply(get_expiration_date, axis = 1)
    
    
    population  = {}
    for date in np.unique(dataframe.ArrivalDate):
            population[date] = (sum((dataframe.ArrivalDate <= date) & (dataframe.ExpirationDate > date)))
    

    dataframe['ShelterPop'] = map(lambda(x):population[x], dataframe.ArrivalDate.tolist())
    
    return dataframe['ShelterPop']
