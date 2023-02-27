# Part 1.2 Cross-validation (using Gini)
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inline as inline
from sklearn.model_selection import cross_val_score
import pydotplus
from IPython.display import Image
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pandas.api.types import is_numeric_dtype
from pandas.plotting import parallel_coordinates
from sklearn import svm
from sklearn import preprocessing

# Read the csv file
from sklearn.metrics import accuracy_score, precision_score

data = pd.read_csv('german.csv', header='infer')

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

PreProcessing()

X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

lb = preprocessing.LabelBinarizer()
y_train = np.array([number[0] for number in lb.fit_transform(y_train)])

classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf = classifier.fit(X_train, y_train)
# clf = clf.fit(X_train, y_train)

Accuracy_Values=cross_val_score(clf, X , y, cv=10, scoring='accuracy')
# print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

precision_Values=cross_val_score(clf, X , y, cv=10, scoring='precision')
# print('\nPrecision values for 10-fold Cross Validation:\n',precision_Values)
print('\nFinal Average precision of the model:', round(precision_Values.mean(),2))

Fmeasure_Values=cross_val_score(clf, X , y, cv=10, scoring='f1_weighted')
# print('\nF-measure values for 10-fold Cross Validation:\n',Fmeasure_Values)
print('\nFinal Average F-measure of the model:', round(Fmeasure_Values.mean(),2))

#Print out table
fig, ax =plt.subplots(1,1)
temp=[Accuracy_Values,precision_Values,Fmeasure_Values]
column_labels=['pass1','pass2','pass3','pass4','pass5','pass6','pass7','pass8','pass9','pass10']
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=temp,colLabels=column_labels,rowLabels=['Accuracy', 'Precision', 'F-measure'])
table.set_fontsize(55)
table.scale(5, 5)
plt.axis('off')
plt.show()

#Print out bar graph
objects = ['Accuracy', 'Precision', 'F-measure']
y_pos = np.arange(len(objects))
performance = [round(Accuracy_Values.mean(),2),round(precision_Values.mean(),2),round(Fmeasure_Values.mean(),2)]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.show()