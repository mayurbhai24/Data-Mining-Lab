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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.api.types import is_numeric_dtype
from pandas.plotting import parallel_coordinates

train_data = pd.read_csv('datingData_training.csv', header='infer')
test_data = pd.read_csv('datingData_test.csv', header='infer')

def PreProcessing():
    test_data['peopletype'].replace({'didntLike':0, 'smallDoses':1, 'largeDoses':2}, inplace=True)
    train_data['peopletype'].replace({'didntLike':0, 'smallDoses':1, 'largeDoses':2}, inplace=True)

def KNearestNeighbor():
    start = time.time()
    X_train = train_data.drop('peopletype', axis=1)
    y_train = train_data['peopletype']
    X_test =  test_data.drop('peopletype', axis=1)
    y_test = test_data['peopletype']

    classifier = KNeighborsClassifier(n_neighbors=5)
    KNN = classifier.fit(X_train, y_train)
    prediction = KNN.predict(X_test)

    FMScore=f1_score(y_test, prediction, average='weighted')
    AccScore = accuracy_score(y_test, prediction)
    PrecScore = precision_score(y_test, prediction, labels=[1,2], average='weighted')

    FMValidation = cross_val_score(classifier, X_test, y_test, cv=10, scoring='f1_weighted')
    FMValidation = round(FMValidation.mean(),5)

    finish = time.time()
    elapsed_Time = (finish-start)
    return FMScore, AccScore, PrecScore, FMValidation, elapsed_Time

def TreeModel ():
    start = time.time()
    X_train = train_data.drop('peopletype', axis=1)
    y_train = train_data['peopletype']
    X_test =  test_data.drop('peopletype', axis=1)
    y_test = test_data['peopletype']

    classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    DT = classifier.fit(X_train, y_train)
    prediction = DT.predict(X_test)

    FMScore=f1_score(y_test, prediction, average='weighted')
    AccScore = accuracy_score(y_test, prediction)
    PrecScore = precision_score(y_test, prediction, labels=[1,2], average='weighted')

    FMValidation = cross_val_score(classifier, X_test, y_test, cv=10, scoring='f1_weighted')
    FMValidation = round(FMValidation.mean(),5)

    finish = time.time()
    elapsed_Time = (finish-start)
    return FMScore, AccScore, PrecScore, FMValidation, elapsed_Time

def NaiveBayes():
    start = time.time()
    X_train = train_data.drop('peopletype', axis=1)
    y_train = train_data['peopletype']
    X_test =  test_data.drop('peopletype', axis=1)
    y_test = test_data['peopletype']

    classifier = GaussianNB()
    NB=classifier.fit(X_train,y_train)
    prediction=NB.predict(X_test)

    FMScore=f1_score(y_test, prediction, average='weighted')
    AccScore = accuracy_score(y_test, prediction)
    PrecScore = precision_score(y_test, prediction, labels=[1,2], average='weighted')

    FMValidation = cross_val_score(classifier, X_test, y_test, cv=10, scoring='f1_weighted')
    FMValidation = round(FMValidation.mean(),5)

    finish = time.time()
    elapsed_Time = (finish-start)
    return FMScore, AccScore, PrecScore, FMValidation, elapsed_Time


PreProcessing()
FMeasureAvg = 0
AccuracyAvg = 0
PrecisionAvg = 0
TimeAvg = 0
FMValidationAvg = 0
print("Running KNN Model")
for i in range(0,5):
    v, w, x, y, z = KNearestNeighbor()
    FMeasureAvg = v + FMeasureAvg
    AccuracyAvg = w + AccuracyAvg
    PrecisionAvg = x +PrecisionAvg
    FMValidationAvg = y
    TimeAvg = z + TimeAvg

FMeasureAvg = FMeasureAvg/5
AccuracyAvg = AccuracyAvg/5
PrecisionAvg = PrecisionAvg/5
TimeAvg = TimeAvg/5
#Print the averages
print('KNN Stats:')
print('F-Measure Average: %.5f' %(FMeasureAvg))
print('Accuracy Average: %.5f' %(AccuracyAvg))
print('Precision Average: %.5f' %(PrecisionAvg))
print('F-Measure Validation Average: ' + str(FMValidationAvg))
print('Time Average: %.5f' %(TimeAvg))
print('---------------------------------------')
print()

FMeasureAvg = 0
AccuracyAvg = 0
PrecisionAvg = 0
TimeAvg = 0
FMValidationAvg = 0
print("Running Tree Decision Model")
for i in range(0,5):
    v, w, x, y, z = TreeModel()
    FMeasureAvg = v + FMeasureAvg
    AccuracyAvg = w + AccuracyAvg
    PrecisionAvg = x + PrecisionAvg
    FMValidationAvg = y
    TimeAvg = z + TimeAvg

FMeasureAvg = FMeasureAvg/5
AccuracyAvg = AccuracyAvg/5
PrecisionAvg = PrecisionAvg/5
TimeAvg = TimeAvg/5
#Print the averages
print('TD Stats:')
print('F-Measure Average: %.5f' %(FMeasureAvg))
print('Accuracy Average: %.5f' %(AccuracyAvg))
print('Precision Average: %.5f' %(PrecisionAvg))
print('F-Measure Validation Average: ' + str(FMValidationAvg))
print('Time Average: %.5f' %(TimeAvg))
print('---------------------------------------')
print()


PreProcessing()
FMeasureAvg = 0
AccuracyAvg = 0
PrecisionAvg = 0
TimeAvg = 0
FMValidationAvg = 0
print("Running Naive Bayes Model")
for i in range(0,5):
    v, w, x, y, z = NaiveBayes()
    FMeasureAvg = v + FMeasureAvg
    AccuracyAvg = w + AccuracyAvg
    PrecisionAvg = x +PrecisionAvg
    FMValidationAvg = y
    TimeAvg = z + TimeAvg

FMeasureAvg = FMeasureAvg/5
AccuracyAvg = AccuracyAvg/5
PrecisionAvg = PrecisionAvg/5
TimeAvg = TimeAvg/5
#Print the averages
print('NB Stats:')
print('F-Measure Average: %.5f' %(FMeasureAvg))
print('Accuracy Average: %.5f' %(AccuracyAvg))
print('Precision Average: %.5f' %(PrecisionAvg))
print('F-Measure Validation Average: ' + str(FMValidationAvg))
print('Time Average: %.5f' %(TimeAvg))
print('---------------------------------------')
print()