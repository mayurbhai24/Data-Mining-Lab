import matplotlib
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inline as inline
import pydotplus
from IPython.display import Image
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.api.types import is_numeric_dtype
from pandas.plotting import parallel_coordinates

#data = pd.read_csv('german.csv', header='infer')
data = pd.read_csv('waveform.csv', header='infer')

def PreProcessing():
    # Prepossessing to remove A
    data['ATT1'].replace({'A11': 1, 'A12': 2, 'A13': 3, 'A14': 4}, inplace=True)
    data['ATT3'].replace({'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4}, inplace=True)
    data['ATT4'].replace(
        {'A40': 0, 'A41': 1, 'A42': 2, 'A43': 3, 'A44': 4, 'A45': 5, 'A46': 6, 'A47': 7, 'A48': 8, 'A49': 9, 'A410': 10},
        inplace=True)
    data['ATT6'].replace({'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4, 'A65': 5}, inplace=True)
    data['ATT7'].replace({'A71': 1, 'A72': 2, 'A73': 3, 'A74': 4, 'A75': 5}, inplace=True)
    data['ATT9'].replace({'A91': 1, 'A92': 2, 'A93': 3, 'A94': 4, 'A95': 5}, inplace=True)
    data['ATT10'].replace({'A101': 1, 'A102': 2, 'A103': 3}, inplace=True)
    data['ATT12'].replace({'A121': 1, 'A122': 2, 'A123': 3, 'A124': 4}, inplace=True)
    data['ATT14'].replace({'A141': 1, 'A142': 2, 'A143': 3}, inplace=True)
    data['ATT15'].replace({'A151': 1, 'A152': 2, 'A153': 3}, inplace=True)
    data['ATT17'].replace({'A171': 1, 'A172': 2, 'A173': 3, 'A174': 4}, inplace=True)
    data['ATT19'].replace({'A191': 0, 'A192': 1}, inplace=True)
    data['ATT20'].replace({'A201': 1, 'A202': 0}, inplace=True)

def KNearestNeighbor(i):
    k = i+1
    X = data.drop('Class', axis=1)
    y = data['Class']

    #Normalization
    PredictorScaler=MinMaxScaler()
    # Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(X)
    # Generating the standardized values of X
    X=PredictorScalerFit.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    classifier = KNeighborsClassifier(n_neighbors=k)
    KNN = classifier.fit(X_train, y_train)
    prediction = KNN.predict(X_test)

    #print(confusion_matrix(y_test, prediction))
    #print(classification_report(y_test, prediction))

    FMScore=f1_score(y_test, prediction, average='weighted')
    #X_train = X_train[:100]
    #y_train = y_train[:100]
    X_train = X_train[:500]
    y_train = y_train[:500]
    FMValidation = cross_val_score(classifier, X_train , y_train, cv=10, scoring='f1_weighted')
    print('K = ' + str(k))
    print('F_Measure of the model on Testing Sample Data: ', round(FMScore,3))
    print('Accuracy of the valdiation: ', round(FMValidation.mean(),3))

    return round(FMValidation.mean(),3)

def TreeModel (i):
    k = i + 1
    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=k)
    classifier.fit(X_train, y_train)

    prediction = classifier.predict(X_test)
    #     print(confusion_matrix(y_test, y_pred))
    #     print(classification_report(y_test, y_pred))

    FMScore=f1_score(y_test, prediction, average='weighted')
    #X_train = X_train[:100]
    #y_train = y_train[:100]
    X_train = X_train[:500]
    y_train = y_train[:500]
    FMValidation = cross_val_score(classifier, X_train , y_train, cv=10, scoring='f1_weighted')
    print('Depth = ' + str(k))
    print('Accuracy of the model on Testing Sample Data:', round(FMScore,3))
    print('Accuracy of the valdiation: ', round(FMValidation.mean(),3))
    return round(FMValidation.mean(),3)

PreProcessing()


FM = []
K = []
print("Running KNN Algorithim")
for i in range(0,5):
    x = KNearestNeighbor(i)
    K.append(i+1)
    FM.append(x)


FM_dt = []
depth = []
print()
print("Running DT Algorithim")
for i in range(2,10):
    x = TreeModel(i)
    depth.append(i+1)
    FM_dt.append(x)

#Plotting KNN and DT side by side
plt.subplot(1, 2, 1)
plt.plot(K,FM,'ro-',label='KNN')
plt.xlabel('K value')
plt.ylabel('F-Measure Accuracy')
plt.subplot(1, 2, 2)
plt.plot(depth,FM_dt,'ro-',label='DT')
plt.xlabel('Depth')
plt.show()
