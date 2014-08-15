import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


import matplotlib.pyplot as plt

data = pd.read_csv('Dogs_Final_Train.csv')


data


Y = data['Status'].get_values()
X = data[data.columns[data.columns!='Status']].get_values()

cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 10)


rf_curve = learning_curve.learning_curve(RandomForestClassifier(200),X,Y,train_sizes = np.array([0.1,0.3,0.5,1]), cv = cross_validation_object)
logistic_curve = learning_curve.learning_curve(LogisticRegression(C = 0.1),X,Y,train_sizes = np.array([0.1,0.3,0.5,1]), cv = cross_validation_object)

fig, axes = plt.subplots(nrows = 2)
axes[0].plot(rf_curve[0], np.mean(rf_curve[1], axis = 1))
axes[0].plot(rf_curve[0], np.mean(rf_curve[2], axis = 1))
axes[0].set_title('Random Forest')
axes[0].set_ylabel('Accuracy: 10-fold CV')

axes[1].plot(logistic_curve[0], np.mean(logistic_curve[1], axis = 1))
axes[1].plot(logistic_curve[0], np.mean(logistic_curve[2], axis = 1))
axes[1].set_title('Logistic Regression')
axes[1].set_ylabel('Accuracy: 10-fold CV')
axes[1].set_xlabel('Number of Data Points')


######## Getting top features from Random Forest
cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 10)
features =data[data.columns[data.columns!='Status']].columns.tolist()

def wereyoutopten(x,features,top_ten_indices):
        location = np.where(np.array(features) == x)
        if location in top_ten_indices[0:10]:
            return 1
        else:
            return 0
ranking=[]
for train,test in cross_validation_object:
    rf_fit = RandomForestClassifier(200)
    rf_fit.fit(X[train], Y[train])
    indices = np.argsort(rf_fit.feature_importances_)[::-1]
    ranking.append(map(lambda(x):wereyoutopten(x, features,indices), features))
    

best_features = [features[i] for i in np.where(np.mean(ranking,axis =0) >0)[0]]

with open('best_features.p','w') as f:
    pk.dump(best_features,f)
#### Using these best features to train

X_new = data[data.columns[data.columns.isin(best_features)]]
Y_new = Y
cross_validation_object = cross_validation.StratifiedKFold(Y, n_folds = 10)

rf_curve = learning_curve.learning_curve(RandomForestClassifier(200),X_new,Y_new,train_sizes = np.array([0.1,0.3,0.5,1]), cv = cross_validation_object)
plt.plot(rf_curve[0],np.mean(rf_curve[1],axis = 1))
plt.plot(rf_curve[0],np.mean(rf_curve[2],axis = 1))
plt.ylim([0.88,1])
plt.title('Random Forest (Best Features)')
plt.ylabel('Accuracy: 10-fold CV')
plt.xlabel('Number of Datapoints')


