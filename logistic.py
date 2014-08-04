#! /Users/jenniferli/Desktop/Python/Pets
import pandas as pd
import numpy as np
from __future__ import division
import matplotlib.pyplot as plt

def percent_adopted(x):
    return sum(x['Status'] == 'ADOPTED')/len(x)

data = pd.read_csv('Dogs_Final_Train.csv')
data.Year = data.Year -2007

size_dict = {'TOY': 4,
    'SMALL':3,
    'MEDIUM':2,
    'LARGE':1,
    'GIANT':5}

age_dict = {'BABY': 2,
    'YOUNG ADULT':4,
    'ADULT':3,
    'SENIOR':1}  
    
 
adopt_dict = {'ADOPTED': 1,
    'EUTHANIZED':0}
   
gender_dict = {'MALE':'MALE',
            'FEMALE':'FEMALE',
            'NEUTER':'FIXED',
            'SPAYED':'FIXED',
            'UNKNOWN':'UNKNOWN',
            'ALTERED':'FIXED'}
            
condition_dict = {'TOO YOUNG': 1,
    'POOR':1,
    'WILD':1,
    'NEEDS MEDICAL':1,
    'FAIR':4,
    'GOOD':5}
    
weekday_dict = {0:0,
    1:1,
    2:1,
    3:1,
    4:2,
    5:1,
    6:1}
            
            
def map_dict(i,mydict):
    if i in mydict.keys():
        return mydict[i]
    else:
        return i


data['Size'] = data['Size'].apply(lambda(x):map_dict(x, size_dict))
data['Age'] = data['Age'].apply(lambda(x):map_dict(x, age_dict))
data['Status'] = data['Status'].apply(lambda(x):map_dict(x, adopt_dict))
data['Gender'] = data['Gender'].apply(lambda(x):map_dict(x, gender_dict))
data['Condition'] = data['Condition'].apply(lambda(x):map_dict(x, condition_dict))
data['Weekday'] = data['Weekday'].apply(lambda(x):map_dict(x, weekday_dict))

## identify dogs of the most common colors
data['BLACK'] = data['Primary Color'].apply(lambda(x):1 if x=='BLACK' else 0)
data['BROWN'] = data['Primary Color'].apply(lambda(x):1 if x=='BROWN' else 0)
data['WHITE'] = data['Primary Color'].apply(lambda(x):1 if x=='WHITE' else 0)
data['TAN'] = data['Primary Color'].apply(lambda(x):1 if x=='TAN' else 0)
data['RED'] = data['Primary Color'].apply(lambda(x):1 if x=='RED' else 0)


## identify dogs of the most breeds
data['RETRIEVER'] = data['Breed'].apply(lambda(x):1 if x=='RETRIEVER' else 0)
data['SHEPHERD'] = data['Breed'].apply(lambda(x):1 if x=='SHEPHERD' else 0)
data['PITBULL'] = data['Breed'].apply(lambda(x):1 if x=='PITBULL' else 0)
data['HEELER'] = data['Breed'].apply(lambda(x):1 if x=='HEELER' else 0)
data['CHIHUAHUA'] = data['Breed'].apply(lambda(x):1 if x=='CHIHUAHUA' else 0)
data['TERRIER'] = data['Breed'].apply(lambda(x):1 if x=='TERRIER' else 0)



def get_null_index(dataframe, column):
    return np.where(dataframe[column].isnull() == True)[0]
    
size_null_index = get_null_index(data, 'Size')
age_null_index = get_null_index(data, 'Age')
gender_null_index = get_null_index(data, 'Gender')

all_null_index = np.unique(list(size_null_index) + list(age_null_index) + list(gender_null_index))

remove_nan_data = data[~data.index.isin(all_null_index)]
remove_nan_data.Month = [1 if x else 0 for x in remove_nan_data.Month.isin([12,2,3])]

del data


from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.learning_curve import validation_curve

Y = remove_nan_data['Status'].get_values()
dummies = pd.get_dummies(remove_nan_data['Gender'])
dummies3 = pd.get_dummies(remove_nan_data['Breed'])
dummies4 = pd.get_dummies(remove_nan_data['Primary Color'])
dummies5 = pd.get_dummies(remove_nan_data['Alphabet'])
dummies6 = pd.get_dummies(remove_nan_data['Arrived As'])
#dummies7 = pd.get_dummies(remove_nan_data['Month'])
dummies8 = pd.get_dummies(remove_nan_data['Arrival Precinct'])


X = pd.concat([remove_nan_data[['Age','Size','Fixed','Has Name','Condition','Year','Month','BLACK','WHITE']],dummies6,dummies5,dummies3,dummies,dummies8],axis = 1).get_values()


### building pipeline
logistic_fit = LogisticRegression()
rf_fit = RandomForestClassifier()
SVC  = SVC(kernel = 'rbf')

remove_low_var = VarianceThreshold()
best_features = SelectKBest(chi2)

pipeline_object = Pipeline([('remove_low_var', remove_low_var),('best_features', best_features),('model', rf_fit)])
pipeline_object = Pipeline([('remove_low_var', remove_low_var),('best_features', best_features),('model', logistic_fit)])


cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 10)
tuned_parameters = [{'model__C': [0.15,0.17],
                    'model__penalty': ['l1'],
                    'remove_low_var__threshold':[0.002],
                    'best_features__k':['all']}]

grid_search_object = GridSearchCV(pipeline_object, tuned_parameters, cv = cross_validation_object)
grid_search_object.fit(X,Y)  # use fit if last item in pipeline is fit.

#
#for train_index,test_index in kfold_indices:
#    X_train, X_test = X[train_index], X[test_index]
#    Y_train, Y_test = Y[train_index], Y[test_index]
#    logistic_fit.fit(X[a], Y[a])
#    
#    
    
#logistic_fit.fit(X[a], Y[a])
#logistic_fit.score(X[b], Y[b])

indices = cross_validation.ShuffleSplit(len(Y), n_iter = 1, test_size = 0.1)
for train_index,test_index in indices:
    a= train_index
    b=test_index
    
best_model = Pipeline([('variance', VarianceThreshold(threshold = 0.002)),('model', LogisticRegression(C = 0.17,penalty = 'l1'))])
best_model.fit(X[a], Y[a])
prediction = best_model.predict(X[b])
ground_truth = Y[b]
wrong = remove_nan_data.iloc[error_index(prediction,ground_truth)]
    


def error_index(x_test, y_test):
    return b[np.where(x_test != y_test)[0]]
    
        

barf = pd.read_csv('Dogs_Final_Test.csv')