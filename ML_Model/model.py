#!/usr/bin/env python
# coding: utf-8

# import libraries
import pandas as pd
import sklearn
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# importing data set
data_set = pd.read_csv('Social_Network_Ads.csv')

# seperating features and label from train data
x = data_set.iloc[:, [2, 3]].values
y = data_set.iloc[:, 4].values

# spliting dataset  into training and test data
x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size=0.25, random_state=0)

# feature scaling
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# fitting classifier  to the training set
classifier = DecisionTreeClassifier(criterion='entropy',  random_state=0)
classifier.fit(x_train, y_train)

# predicting the test set result
y_prediction = classifier.predict(x_test)

# making the confusion matrix
cf_matrix = confusion_matrix(y_test, y_prediction)

# getting accuracy
accuracy = sklearn.metrics.balanced_accuracy_score(y_test,y_prediction)*100
print(accuracy)

# setting  threshold limit for accuracy
# if accuracy is above the limit then make pickle file
if accuracy > 85:
    file = open('TrainPickle.pkl', 'wb')
    pickle.dump(classifier, file)
    file.close()
