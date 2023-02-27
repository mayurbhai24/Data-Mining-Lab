import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from pandas.api.types import is_numeric_dtype

missing_values = [" ?"]
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None, na_values = missing_values)
data.columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 'race', 'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']

# data2 = data['age'] + data['sex']
# print(data2)

#2.1
continuous_attributes = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'captial-loss', 'hours-per-weak']

for col in data.columns:
#     print(col)
    if is_numeric_dtype(data[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data[col].mean())
        print('\t Standard deviation = %.2f' % data[col].std())
        print('\t Minimum = %.2f' % data[col].min())
        print('\t Maximum = %.2f' % data[col].max())

#2.2
discrete_attri=['workclass', 'education', 'martial-status', 'occupation', 'race', 'relationship', 'sex', 'native-country','income']

for i in range(0,9):
    print(data[discrete_attri[i]].value_counts())
    print()

#2.3
data['occupation'].hist(bins=14, figsize=(12,8))
# data.occupation.value_counts().plot(kind='bar')
plt.xticks(rotation='vertical')
plt.show()
# Most people work in prod-speicalty and craft-repair

#2.4
data['age'].hist(bins=15, figsize=(12,8))
# data.occupation.value_counts().plot(kind='bar')
# plt.xticks(rotation='vertical')
plt.show()
# Majority of the poeple are aged around 35 years old and most of them are under 50 years old.

#2.5
plt.plot(data['age'], data['education'], 'o',color='red')
plt.xlabel('age')
plt.ylabel('education')
plt.show()
# Each education level has people in it with variety of different ages. Education level preschool is staggered as
# less poepl are expected to be in the group.

plt.plot(data['capital-gain'], data['age'],'o', color='red')
plt.xlabel('capital-gain')
plt.ylabel('age')
plt.show()
# captial gain of 100000 is observed in age groups upto around 80 years of age. Between 40000 and 100000, there are almost 0
# people with captial gain in that range. Most poeple have a cpatial gain between 0 and 20000.

plt.plot(data['capital-loss'], data['age'],'o', color='red')
plt.xlabel('capital-loss')
plt.ylabel('age')
plt.show()
# lot people with capital loss between 1000 and 3000. Every age group has soemone with 0 captial loss

plt.plot(data['relationship'], data['age'],'o', color='red')
plt.xlabel('relationship')
plt.ylabel('age')
plt.show()
# Lost of poeple in the age group 70 to 90 dont own-child.

plt.plot(data['capital-gain'], data['race'],'o', color='red')
plt.xlabel('capital-gain')
plt.ylabel('race')
plt.show()
#white people have the most diverse range of captial gains. As the population of each group decreases, smaller the range of captial-gain.
# So if you are an 'other' then you are most likely to have a captial gain of 0-20000 or 100000

#2.6
data2 = data[['age','hours-per-week','sex']].copy()
pc1 = parallel_coordinates(data2, 'sex', color=('deepskyblue', 'fuchsia'))
pc1.figure
# Younger the person, less hours they work. A lot more males with high hours-per-week then females.

data3 = data[['capital-gain', 'capital-loss','race']].copy()
pc = parallel_coordinates(data3, 'race', color=('red','deepskyblue', 'fuchsia','lime','black'))
pc.figure
