from __future__ import division

import pandas as pd
import numpy as np
import phonemes

import pickle as pk

import pymysql as mdb
from pandas.io import sql

from sklearn import cross_validation
import scipy.signal

import os.path

from number_of_animals_at_shelter import get_pop

###########


def list_of_current_animals():
    with open('list_of_current_animals.pk','r') as f:
        return pk.load(f)       

def read_in_data():
    ''' reads in the cleaned table from MySQL'''
    con =  mdb.connect('localhost', 'root','***','HomewardBound'); 
    
    with con:
        dogs = sql.read_sql('select * from Pets_cleaned;', con = con)   
        weather = sql.read_sql('select * from PetWeather;',con = con)
        econ = sql.read_sql('select * from PetEcon;',con = con)
        


    return dogs, weather, econ
    
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


adopt_dict = {'ADOPTED': 1,
        'EUTHANIZED':0}
    
gender_dict = {'MALE':'MALE',
                'FEMALE':'FEMALE',
                'NEUTER':'MALE',
                'SPAYED':'FEMALE'}
                
condition_dict = {'GOOD':'GOOD',
                'FAIR':'FAIR',
                'POOR':'BAD',
                'NEEDS MEDICAL':'BAD',
                'TOO YOUNG':'BAD',
                'WILD':'BAD'}
                
                
def map_dict(i,mydict):
        if i in mydict.keys():
            return mydict[i]
        else:
            return i
            
            

class top_fraction(object):
    '''generates two new columns that only contains the most popular breeds and colors'''
    def __init__(self, fraction, input_column = ['Breed', 'PrimaryColor'], return_column = ['TopBreed','TopColor']):
        self.input_column = input_column
        self.fraction = fraction
        self.return_column = return_column
        
    def transform(self,dataframe):
        for index in xrange(len(self.input_column)):
            list_of_labels = np.cumsum(dataframe.groupby(self.input_column[index]).apply(lambda(x): len(x)/len(dataframe)).order(ascending = False))
            desired_labels = list_of_labels[list_of_labels < self.fraction].index
            # make new column
            dataframe[self.return_column[index]] = dataframe[self.input_column[index]].apply(lambda(x): x if x in desired_labels else np.nan)  
        return dataframe


def get_sanitized():
    dogs_final, weather, econ = read_in_data()

    ############
    ## DATE INFO
    ############
    
    ## converting date info to pandas-friendly format
    dogs_final['ArrivalDate'] = pd.to_datetime(dogs_final['ArrivalDate'])
    dogs_final['Year'] = map(lambda(x):x.year, dogs_final['ArrivalDate'].tolist())
    dogs_final['Month'] = map(lambda(x):x.month, dogs_final['ArrivalDate'].tolist())
    dogs_final['Day'] = map(lambda(x):x.day, dogs_final['ArrivalDate'].tolist())

    ####################################
    #  MERGING WITH ECON DATA and weather data
    ########################################

    dogs_final = dogs_final.merge(econ, on = ['Month','Year'])
    

    dogs_final = dogs_final.merge(weather, on = ['Month','Year','Day'])
    
    ##############
    # LENGTH of STaY (number of days)-- convert to int
    ##############
    dogs_final['LengthofStay'] = [int(x) if x is not None else 999 for x in dogs_final['LengthofStay']]
    
    ###################
    ## OVERCROWDING
    ###################
    
    ### figuring out how many dogs arrived within a 10 day period..indicate of how crowded things are
    new_frame = pd.DataFrame(index = pd.date_range(min(dogs_final.ArrivalDate),max(dogs_final.ArrivalDate),freq='D'))
    date_agg = pd.DataFrame(dogs_final.groupby('ArrivalDate').apply(lambda(x):len(x)))
    
    new_frame['Raw'] = [date_agg.ix[i][0] if i in date_agg.index else 0 for i in new_frame.index]
    new_frame['Window'] = scipy.signal.convolve([new_frame['Raw'].tolist()[0] for x in range(10)]+ new_frame['Raw'].tolist(),[1 for i in range(10)])[10:-9]

    def arrived(x):
        return new_frame['Window'].ix[x]
        
    dogs_final['PupInflux'] = dogs_final['ArrivalDate'].apply(arrived)
    dogs_final['ShelterPop'] = get_pop(dogs_final)

        


    ############
    ## NAME INFO
    ############
    
    ## cleaning up name info
    dogs_final['PetName'] = map(pet_name_cleanup , dogs_final['PetName'].tolist())
    
    ## does the animal have a name? 
    dogs_final['HasName'] = map(has_name , dogs_final['PetName'].tolist())
    
        ## getting phonetic info
    dogs_final['Phonetic'] = map(phonetic , dogs_final['PetName'].tolist())

        ## getting alphabet info, and length of name
    dogs_final['Alphabet'] = [np.nan if pd.isnull(x) else x[0] for x in dogs_final['PetName']]
    dogs_final['NameLength'] = [0 if pd.isnull(x) else len(x) for x in dogs_final['PetName']]
    
    ############
    ## BREED INFO
    ############

    dogs_final['Breed'] = dogs_final['Breed'].apply(breed)
    
    #############################
    ## which rows have NaNs or Unknowns for Gender, Pet Condition, Age, and Size?
    #######################
    remove_indices = np.unique(list(np.where(dogs_final['Gender'].isnull())[0])+\
    list(np.where(dogs_final['PetCondition'].isnull())[0])+\
    list(np.where(dogs_final['Age'].isnull())[0])+\
    list(np.where(dogs_final['Size'].isnull())[0])+\
    list(np.where(dogs_final['Gender'].isin(['ALTERED','UNKNOWN']))[0]))

    ##################################
    # FILLING IN NaN with most common
    ################################
    
    #dogs_final[np.where(dogs_final['Gender'].isnull())] = dogs_final.groupby('Gender').size().index[0]
    #dogs_final[np.where(dogs_final['Age'].isnull())] = dogs_final.groupby('Age').size().index[0]
    #dogs_final[np.where(dogs_final['PetCondition'].isnull())] = dogs_final.groupby('PetCondition').size().index[0]

    ####################
    # MAKING DUMMY VARIABLES
    ######################
    instance = top_fraction(0.8)
    dogs_final = instance.transform(dogs_final)
    
    dogs_final['Status'] = dogs_final['Status'].apply(lambda(x):map_dict(x, adopt_dict))
    dogs_final['Gender'] = dogs_final['Gender'].apply(lambda(x):map_dict(x, gender_dict))
    dogs_final['PetCondition'] = dogs_final['PetCondition'].apply(lambda(x):map_dict(x, condition_dict))
    
    dummies = pd.get_dummies(dogs_final['Gender']).iloc[:,[1]]
    dummies2 = pd.get_dummies(dogs_final['TopBreed'])
    dummies3 = pd.get_dummies(dogs_final['TopColor'])
    dummies4 = pd.get_dummies(dogs_final['ArrivedAs']).iloc[:,:-1]
    dummies5 = pd.get_dummies(dogs_final['Age']).iloc[:,:-1]
    dummies6 = pd.get_dummies(dogs_final['Size']).iloc[:,:-1]
    dummies7 = pd.get_dummies(dogs_final['PetCondition']).iloc[:,:-1]
    
    largedata = pd.concat([dogs_final[['ArrivalDate','AnimalID','Status','Fixed','HasName','PupInflux','HEAT','RAIN','Unemployment','Population','LengthofStay']],dummies,dummies2,dummies3,dummies4,dummies5,dummies6, dummies7],axis = 1)

    #### getting examples from website
    
    example_data = largedata[largedata['AnimalID'].isin(list_of_current_animals())]
    example_data = example_data.groupby('AnimalID').apply(lambda(x):x.sort('ArrivalDate').iloc[-1])    # grab most recent
    example_data.to_csv('Dogs_Final_DEMO.csv',index = False)

    ################################
    ## generating data for the model
    ################################
    modeldata = largedata[~largedata.index.isin(remove_indices)]

    modeldata = modeldata[modeldata['ArrivalDate'] < min(example_data.ArrivalDate) -  pd.DateOffset(days = 60)]  ### 

    modeldata = modeldata[modeldata.Status.isin([1,0])]
    
    ############
    # PARSING out TRAINING/TESTING DATA, then saving as .csv
    ##################
    
    if os.path.isfile('cross_validation.pk'):
        with open('cross_validation.pk','rb') as f:
            indices = pk.load(f)
    else:
        indices = cross_validation.ShuffleSplit(len(modeldata),n_iter = 1,test_size = 0.25)
        with open('cross_validation.pk','wb') as f:
            pk.dump(indices, f)
            
    

    for train,test in indices:
        tr = train
        ts = test
        
    ## training set:
    modeldata.iloc[tr].to_csv('Dogs_Final_Train.csv',index = False)
    
    
    ## test set:
    modeldata.iloc[ts].to_csv('Dogs_Final_Test.csv', index = False)


if __name__ == '__main__':
    get_sanitized()
    


    

