from __future__ import division
import pandas as pd
import numpy as np
import phonemes
import pickle as pk
from sklearn import cross_validation

def read_in_data(filename):
    ''' reads in the data and selects out the desired columns'''
    data = pd.read_csv(filename)
    dogs=data[(data['Animal Type'] =='DOG') & ((data['Status'] == 'ADOPTED')|(data['Status'] == 'EUTHANIZED')) & \
    ((data['Arrived As'] == 'STRAY')|(data['Arrived As'] == 'UNABLE TO CARE FOR')|(data['Arrived As'] == 'UNWANTED'))]  
    desired_columns = ['Arrival Date', 'Arrived As', 'Breed', 'Gender', 'Pet Name', 'Status', 'Condition', 'Arrival Precinct','Length of Stay','Fixed', 'Age','Size','Primary Color','Secondary Color']
    
    dogs.index = range(len(dogs))
    return dogs[desired_columns]
    
def adopt_euth_ratio(x):
    return sum(x['Status'] == 'ADOPTED')/sum(x['Status'] == 'EUTHANIZED')

def pet_name_cleanup(x):
    ''' removes non-letter characters from pet name'''
    temp_val = np.nan if pd.isnull(x) else x.strip('[0-9] .?/*#^-_!')
    if temp_val == '':
        temp_val = np.nan
    return temp_val
    

def has_name(x):
    ''' does the animal have a name? 0 if no, 1 if yes'''
    return 0 if pd.isnull(x) else 1
    
def phonetic(x):
    return np.nan if pd.isnull(x) else phonemes.soundex(x)[1:]


def breed(x):
    with open('breed_dictionary.pik','r') as f:
        breed_dictionary = pk.load(f) 
    
    if x in breed_dictionary.keys():
        return breed_dictionary[x]
    else:
        return x
        
def remove_nans(dataframe, column):
    for i in column:
        dataframe = dataframe[~dataframe[i].isnull()]
    return dataframe



def main():
    dogs_final = read_in_data('Pets.csv')

    ############
    ## DATE INFO
    ############
    
    ## converting date info to pandas-friendly format
    dogs_final['Arrival Date'] = pd.to_datetime(dogs_final['Arrival Date'])
    dogs_final['Month'] = map(lambda(x):x.month, dogs_final['Arrival Date'].tolist())
    dogs_final['Weekday'] = map(lambda(x):x.dayofweek, dogs_final['Arrival Date'].tolist())
    dogs_final['Year'] = map(lambda(x):x.year, dogs_final['Arrival Date'].tolist())

    ############
    ## NAME INFO
    ############
    
    ## cleaning up name info
    dogs_final['Pet Name'] = map(pet_name_cleanup , dogs_final['Pet Name'].tolist())
    
    ## does the animal have a name? 
    dogs_final['Has Name'] = map(has_name , dogs_final['Pet Name'].tolist())
    
        ## getting phonetic info
    dogs_final['Phonetic'] = map(phonetic , dogs_final['Pet Name'].tolist())

        ## getting alphabet info, and length of name
    dogs_final['Alphabet'] = [np.nan if pd.isnull(x) else x[0] for x in dogs_final['Pet Name']]
    dogs_final['Name Length'] = [0 if pd.isnull(x) else len(x) for x in dogs_final['Pet Name']]
    
    ############
    ## BREED INFO
    ############

    dogs_final['Breed'] = dogs_final['Breed'].apply(breed)
    
    
    ###########
    # REMOVING DEAD ANIMALS
    #############
    dogs_final = dogs_final[~dogs_final['Condition'].isin(['DEAD ON ARRIVAL'])]
    
    ###########
    # REMOVING the 2 dogs from 2000
    #############
    dogs_final = dogs_final[~dogs_final['Year'].isin([2000])]
    
    ############
    # REMOVING SOME MISSING DATA
    ##################
    bark = remove_nans(dogs_final,['Size','Age','Gender'])
    
    ############
    # PARSING out TRAINING/TESTING DATA, then saving in separate files
    ##################
    
    indices = cross_validation.ShuffleSplit(len(dogs_final),n_iter = 1,test_size = 0.25)

    for train,test in indices:
        tr = train
        ts = test
        
    ## training set:
    dogs_final.iloc[tr].to_csv('Dogs_Final_Train.csv')
    
    ## test set:
    dogs_final.iloc[ts].to_csv('Dogs_Final_Test.csv')


    

main()


    

