import pickle as pk
import re
import pandas as pd
import numpy as np

x = pd.read_csv('Pets.csv')

def collect_breed(dataframe):
    corgi = [x[0] for x in filter(None, map(lambda(x): re.findall('.*CORGI.*',x), np.unique(dataframe['Breed'])))]
    pitbull = [x[0] for x in filter(None, map(lambda(x): re.findall('.*PIT.*BULL.*',x), np.unique(dataframe['Breed'])))]
    dalmation = [x[0] for x in filter(None, map(lambda(x): re.findall('.*DALMAT.*',x), np.unique(dataframe['Breed'])))]
    poodle = [x[0] for x in filter(None, map(lambda(x): re.findall('.*POODLE.*',x), np.unique(dataframe['Breed'])))]
    retriever = [x[0] for x in filter(None, map(lambda(x): re.findall('RETREIVER',x), np.unique(dataframe['Breed'])))]
    
    breed_dictionary = {}
    
    
    def replace(x,y):
        for i in x:
            breed_dictionary[i] = y
            
    replace(corgi, 'CORGI')
    replace(pitbull, 'PITBULL')
    replace(dalmation, 'DALMATIAN')
    replace(poodle, 'POODLE')
    replace(retriever, 'RETRIEVER')
    
    return breed_dictionary
    
def main():
    breed_dictionary =  collect_breed(x)
    with open('breed_dictionary.pik','wb') as f:
        pk.dump(breed_dictionary,f,-1)    
      

