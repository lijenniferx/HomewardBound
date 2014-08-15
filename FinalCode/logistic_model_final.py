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
#top_ten_coefs = grid_search_object.best_estimator_.named_steps['model'].coef_[0][largest_coefficients][0:10]
#top_ten_labels = column_names[largest_coefficients][0:10]
#indices = top_ten_coefs.argsort()
#import string
#for i in range(10):
#    if top_ten_coefs[indices[i]] > 0:
#        my_color ='r'
#    else:
#        my_color = 'k'
#    plt.bar(i, top_ten_coefs[indices[i]],color = my_color)
#    plt.text(i + 0.4,-5,string.upper(top_ten_labels[indices[i]]), rotation = 90, color = my_color)
#plt.gcf().subplots_adjust(bottom=0.30)
#plt.ylabel('Regression coefficient')
#plt.xticks(np.arange(0.5,10.5,1), [],rotation=90)
#plt.grid()

#pvalues.argsort()
with open('finalmodel_train.pk','w') as f:
    pk.dump(grid_search_object.best_estimator_,f)

##### checking on test data

X_test,Y_test, blah = prepare_for_model('Dogs_Final_Test.csv')

with open('finalmodel_train.pk','r') as f:
    finalmodel_train = pk.load(f)
    
finalmodel_train.score(X_test,Y_test)


##### combining training and test data to generate model:

X_test,Y_test, blah = prepare_for_model('Dogs_Final_Test.csv')
X,Y,column_names = prepare_for_model('Dogs_Final_Train.csv')

X_all = np.vstack((X,X_test))
Y_all = np.hstack((Y,Y_test))
with open('finalmodel_train.pk','r') as f:
    finalmodel = pk.load(f)

finalmodel.fit(X_test,Y_test)

with open('finalmodel.pk','w') as f:
    pk.dump(finalmodel,f)