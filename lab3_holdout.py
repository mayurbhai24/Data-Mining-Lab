# Part 1.1 Holdout (using Gini)
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inline as inline
import pydotplus
from IPython.display import Image
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pandas.api.types import is_numeric_dtype
from pandas.plotting import parallel_coordinates

# Read the csv file
from sklearn.metrics import accuracy_score, precision_score

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

def TreeModel (i):
    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=5)
    classifier.fit(X_train, y_train)

    '''
    if i ==0:
        dot_data = tree.export_graphviz(classifier, feature_names=X.columns, class_names=['0','1','2'], filled=True, out_file=None)
        print("test")
        graph = pydotplus.graph_from_dot_data(dot_data)
        Image(graph.create_png())
        graph.write_png('tree' + str(i) + '.png')

    '''
    y_pred = classifier.predict(X_test)
    #     print(confusion_matrix(y_test, y_pred))
    #     print(classification_report(y_test, y_pred))

    AccScore = accuracy_score(y_test, y_pred)
    #     print('Accuracy: %.2f' %(AccScore))
    PrecScore = precision_score(y_test, y_pred, labels=[1,2], average='weighted')
    #     print('Precision: %.2f' %(PrecScore))
    FMScore = f1_score(y_test, y_pred, average='weighted')
    #     print('F-Measure: %.2f' %(FMScore))
    return AccScore, PrecScore, FMScore


#PreProcessing()

#Compute the Average
AccuracyAvg = 0
PrecisionAvg = 0
FMeasureAvg = 0
acc_table = []
ps_table = []
f1_table = []
for i in range(0,5):
    x, y, z = TreeModel(i)
    AccuracyAvg = x + AccuracyAvg
    PrecisionAvg = y + PrecisionAvg
    FMeasureAvg = z +FMeasureAvg
    #To create table
    acc_table.append(x)
    ps_table.append(y)
    f1_table.append(z)

AccuracyAvg = AccuracyAvg/5
PrecisionAvg = PrecisionAvg/5
FMeasureAvg = FMeasureAvg/5

#Print the averages
print()
print('Accuracy Average: %.2f' %(AccuracyAvg))
print('Precision Average: %.2f' %(PrecisionAvg))
print('F-Measure Average: %.2f' %(FMeasureAvg))

#Print out table
fig, ax =plt.subplots(1,1)
temp=[acc_table,ps_table,f1_table]
# print(temp)
column_labels=['pass1','pass2','pass3','pass4','pass5']
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=temp,colLabels=column_labels,rowLabels=['Accuracy', 'Precision', 'F-measure'])
table.set_fontsize(20)
table.scale(2, 2)
plt.show()

#Print out bar graph
objects = ['Accuracy', 'Precision', 'F-measure']
y_pos = np.arange(len(objects))
performance = [AccuracyAvg,PrecisionAvg,FMeasureAvg]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.show()