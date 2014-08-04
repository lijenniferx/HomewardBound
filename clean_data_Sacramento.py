def has_name(x):
    ''' does the animal have a name? 0 if no, 1 if yes'''
    return 0 if pd.isnull(x) else 1


import pandas as pd
sac_dogs = pd.read_csv('Pet_data_Sacramento.csv')
sac_dogs = sac_dogs[sac_dogs['Animal Type'] == 'DOG']
sac_dogs = sac_dogs[sac_dogs['Outcome'].isin(['ADOPTION','EUTH','EUTH VET'])]
sac_dogs['Has Name'] = sac_dogs['Animal Name'].apply(has_name)

