import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from get_model_columns import get_columns_poisson
import statsmodels.api as sm


def mean_squared_error(pred,ground_truth):
    return np.mean((pred - ground_truth)**2)
    

def prepare_for_model(filename,isadopted):
    data = pd.read_csv(filename)

    Y = data['LengthofStay'].get_values()
    X, column_names = get_columns_poisson(data)
    

    relevant_indices = (np.where(X[:,0]==isadopted))[0]
    X = X[relevant_indices,:]
    Y = Y[relevant_indices]

    
    return X[:,1:],Y,data.AnimalID[relevant_indices]
    
X,Y,junk = prepare_for_model('Dogs_Final_Train.csv',1)


scaler = MinMaxScaler([0,1])
X = scaler.fit_transform(X)

Y = np.array([30 if i > 30 else i for i in Y])

##### using cross validation to choose a model

cross_validation_object  = cross_validation.StratifiedKFold(Y, 10)

#### assessing performance of the poisson regression model
performance_poisson = []
for x in [0.01,0.1,1,5,10]:
    cost = []
    for a,b in cross_validation_object:
        resultingmodel = sm.Poisson(Y[a],X[a])
        #res = resultingmodel.fit(disp=False, maxiter = 200)
        res2 = resultingmodel.fit_regularized( alpha = x, maxiter = 200)
        cost.append(mean_squared_error(res2.predict(X[b]), Y[b]))
    performance_poisson.append(np.mean(cost))
    
#### assessing performance of the negative binomial regression model
performance_negativebinomial = []
for x in [0.01,0.1,1,5,10]:
    cost = []
    for a,b in cross_validation_object:
        resultingmodel = sm.NegativeBinomial(Y[a],X[a],loglike_method = 'geometric')
        #res = resultingmodel.fit(disp=False, maxiter = 200)
        res2 = resultingmodel.fit_regularized(alpha = x, maxiter = 200)
        cost.append(mean_squared_error(res2.predict(X[b]), Y[b]))
    performance_negativebinomial.append(np.mean(cost))


##### Log linear model ########## not even close. 
from sklearn.linear_model import ElasticNetCV
linear_fit = ElasticNetCV(cv = cross_validation_object, alphas = [0.01,0.1,1,5,10])
linear_fit.fit(X,np.log(Y+1))
mean_squared_error(np.exp(linear_fit.predict(X)) - 1, Y)


########## creating final model using train data + test data


X_test,Y_test,junk = prepare_for_model('Dogs_Final_Test.csv',1)
X,Y,junk = prepare_for_model('Dogs_Final_Train.csv',1)
scaler = MinMaxScaler([0,1])
X_all = scaler.fit_transform(np.vstack((X_test,X)))
Y_all = np.hstack((Y_test,Y))
Y_all = np.array([30 if i > 30 else i for i in Y_all])
final_model = sm.NegativeBinomial(Y_all,X_all,loglike_method = 'geometric')
res2 = final_model.fit_regularized( alpha = 5, maxiter = 200)


#### fitting final model on demo data
X_demo,Y_demo,animalid = prepare_for_model('Dogs_Final_DEMO.csv',1)
X_demo = scaler.transform(X_demo)
predictions = res2.predict(X_demo)


import pymysql as mdb
con =  mdb.connect('localhost', 'root','***','HomewardBound'); 
with con:
        cur = con.cursor()
        cur.execute("USE HomewardBound")
        for i in xrange(len(animalid)):
            cur.execute('''INSERT INTO Demo_Critter_Days VALUES (%d,%d)''' % (animalid.iloc[i], predictions[i]))
            
            

###############
##### plotting 
################
#X,Y,junk = prepare_for_model('Dogs_Final_Train.csv',1)
#scaler = MinMaxScaler([0,1])
#X = scaler.fit_transform(X)
#Y = np.array([30 if i > 30 else i for i in Y])
#
#final_model = sm.NegativeBinomial(Y,X,loglike_method = 'geometric')
#res2 = final_model.fit_regularized(maxiter =200,alpha =10)
#
#
#X_test,Y_test,junk = prepare_for_model('Dogs_Final_Test.csv',1)
#X_test = scaler.transform(X_test)
#
#
#
#
#import matplotlib.pylab as plt
#xedges = range(50)
#yedges = range(50)
#H, xedges, yedges = np.histogram2d(res2.predict(X_test),Y_test,bins=(xedges, yedges))
#im = plt.imshow(H, interpolation='nearest', origin='low')
#im.set_clim(0,20)
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.title('Number of Days Until Adoption')
#
#
##fig,axes = plt.subplots(nrows = 2)
##axes[0].hist(adopted[adopted.BABY == 1].LengthofStay.get_values(),range(100))
##p = 1 - np.mean(adopted[adopted.BABY == 1].LengthofStay)/ (1 + np.mean(adopted[adopted.BABY == 0].LengthofStay))
##axes[1].hist(scipy.stats.nbinom.rvs(1,p,size = len(adopted)),range(100))
#
#
