import pandas as pd
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from MySQLPets import csv_to_mysql

workforce = data = pd.read_csv('Workforce.csv')
unemployment = data = pd.read_csv('Unemployment.csv')
population = data = pd.read_csv('Population.csv')


workforce['Date'] = pd.to_datetime(workforce['Date'],format = '%Y-%m-%d')
unemployment['Date'] = pd.to_datetime(unemployment['Date'])
population['Date'] = pd.to_datetime(population['Date'],format = '%Y-%m-%d')


### joining tables together

economics = workforce.merge(unemployment)

economics['Month'] = economics['Date'].apply(lambda(x):x.month)
economics['Year'] = economics['Date'].apply(lambda(x):x.year)

population['Month'] = population['Date'].apply(lambda(x):x.month)
population['Year'] = population['Date'].apply(lambda(x):x.year)

econ_final = pd.merge(economics, population, how = 'outer')
econ_final = econ_final.fillna(method = 'ffill')

econ_final.to_csv('PetEcon.csv',index = False)


csv_to_mysql('PetEcon.csv', 'PetEcon')

