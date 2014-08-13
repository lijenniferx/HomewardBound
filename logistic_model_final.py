import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
import pickle as pk
import matplotlib.pyplot as plt

from get_model_columns import get_columns

def prepare_for_model(filename):
    data = pd.read_csv(filename)

    
    Y = data['Status'].get_values()
    X, column_names = get_columns(data)
    
    return X,Y,column_names
    
X,Y,column_names = prepare_for_model('Dogs_Final_Train.csv')


#scores, pvalues = chi2(X, Y)
#selector = RFE(logistic_fit, step =1)

cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 10)
scaler = MinMaxScaler(feature_range = [0,1])
logistic_fit = LogisticRegression()

pipeline_object = Pipeline([('scaler', scaler),('model', logistic_fit)])

tuned_parameters = [{'model__C': [0.01,0.1,1,10],
                    'model__penalty': ['l1','l2']}]

grid_search_object = GridSearchCV(pipeline_object, tuned_parameters, cv = cross_validation_object, scoring = 'accuracy')

grid_search_object.fit(X,Y)  # use fit if last item in pipeline is fit.

#grid_search_object.best_estimator_.fit(X,Y)
#
#largest_coefficients =  abs(grid_search_object.best_estimator_.named_steps['model'].coef_).argsort()[0][::-1]
#
#pvalues.argsort()
with open('finalmodel.pk','w') as f:
    pk.dump(grid_search_object.best_estimator_,f)

##### checking on test data

X_test,Y_test, blah = prepare_for_model('Dogs_Final_Test.csv')

with open('finalmodel.pk','r') as f:
    finalmodel = pk.load(f)
    
finalmodel.score(X_test,Y_test)
