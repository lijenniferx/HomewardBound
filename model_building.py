import pandas as pd
import numpy as np
from __future__ import division
import matplotlib.pyplot as plt


def percent_adopted(x):
    return sum(x['Status'] == 'ADOPTED')/len(x)

def data_preprocessing(filename):
    


    data = pd.read_csv(filename)


    adopt_dict = {'ADOPTED': 1,
        'EUTHANIZED':0}
    
    gender_dict = {'MALE':'MALE',
                'FEMALE':'FEMALE',
                'NEUTER':'MALE',
                'SPAYED':'FEMALE',
                'UNKNOWN':'UNKNOWN',
                'ALTERED':'FIXED'}
                
    def map_dict(i,mydict):
        if i in mydict.keys():
            return mydict[i]
        else:
            return i
            
    def top_fraction(dataframe, input_column,fraction, return_column):
        for index in xrange(len(input_column)):
            list_of_labels = np.cumsum(dataframe.groupby(input_column[index]).apply(lambda(x): len(x)/len(dataframe)).order(ascending = False))
            desired_labels = list_of_labels[list_of_labels < fraction].index
            # make new column
            dataframe[return_column[index]] = dataframe[input_column[index]].apply(lambda(x): x if x in desired_labels else np.nan)  
        return dataframe
    
    #### data processing
    data.Year = data.Year -2007
    data['Status'] = data['Status'].apply(lambda(x):map_dict(x, adopt_dict))
    data['Gender'] = data['Gender'].apply(lambda(x):map_dict(x, gender_dict))
    data['Has Secondary'] = data['Secondary Color'].apply(lambda(x):1 if x else 0)
    
    data = top_fraction(data, ['Breed', 'Primary Color'], 0.8, ['Top Breed','Top Color'])
    
    return data


data = data_preprocessing('Dogs_Final_Train.csv')
#### model building
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.learning_curve import validation_curve



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn import learning_curve
from sklearn.pipeline import Pipeline


def error_index(x_test, y_test):
    return b[np.where(x_test != y_test)[0]]
    
Y = data['Status'].get_values()
dummies = pd.get_dummies(data['Gender'])
dummies2 = pd.get_dummies(data['Top Breed'])
dummies3 = pd.get_dummies(data['Top Color'])
dummies4 = pd.get_dummies(data['Arrived As'])
dummies5 = pd.get_dummies(data['Arrival Precinct'])
dummies6 = pd.get_dummies(data['Age'])
dummies7 = pd.get_dummies(data['Size'])
dummies8 = pd.get_dummies(data['Condition'])
dummies9 = pd.get_dummies(data['Month'])

X = pd.concat([data[['Fixed','Has Name','Year','Has Secondary']],dummies,dummies2,dummies3,dummies4,dummies5,dummies6,dummies7, dummies8,dummies9],axis = 1).get_values()

rf_fit = RandomForestClassifier()

logistic_fit = LogisticRegression()

remove_low_var = VarianceThreshold()
best_features = SelectKBest(chi2)

pipeline_object = Pipeline([('remove_low_var', remove_low_var),('best_features', best_features),('model', rf_fit)])
#pipeline_object = Pipeline([('remove_low_var', remove_low_var),('best_features', best_features),('model', logistic_fit)])

cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 10)
tuned_parameters = [{'model__n_estimators':[200],
                     'model__max_features':[0.1, 0.3],
                    #'model__C': [0.15,0.17],
                    #'model__penalty': ['l1','l2'],
                    'remove_low_var__threshold':[0.000],
                    'best_features__k':['all']}]

grid_search_object = GridSearchCV(pipeline_object, tuned_parameters, cv = cross_validation_object)
grid_search_object.fit(X,Y)  # use fit if last item in pipeline is fit.


##### finding top features

indices = cross_validation.ShuffleSplit(len(Y), n_iter = 1, test_size = 0.25)
for train_index,test_index in indices:
    a= train_index
    b=test_index
    
#best_model = Pipeline([('variance', VarianceThreshold(threshold = 0.000)),('model', LogisticRegression(C = 0.15,penalty = 'l1'))])
best_model = Pipeline([('variance', VarianceThreshold(threshold = 0.000)),('model', RandomForestClassifier(n_estimators = 200,max_features = 0.1))])
best_model.fit(X[a], Y[a])
prediction = best_model.predict(X[b])
ground_truth = Y[b]
wrong = data.iloc[error_index(prediction,ground_truth)]

def plot_errors(column):
    barf = data.groupby(column).apply(lambda(x):len(x)/len(data))
    barf2 = wrong.groupby(column).apply(lambda(x):len(x)/len(wrong))


    for i in wrong.groupby(column).size().index:
        plt.plot(barf[i],barf2[i],'ro')
        plt.text(barf[i] + np.random.normal(0,scale = 0.001,size = 1),barf2[i] + np.random.normal(0,scale = 0.001,size = 1),i,size = 10)
        plt.plot([0,0.3],[0,0.3])
        plt.grid()

conv(data[data['Breed'] == 'BORDER COLLIE']['Status'].tolist(),window[0])

from scipy.signal import convolve as conv
window = np.ones((1,20))

ahem = pd.DataFrame()
ahem['a']=data[data['Breed'] == 'BORDER COLLIE']['Arrival Date']
ahem['b']=data[data['Breed'] == 'BORDER COLLIE']['Status']
