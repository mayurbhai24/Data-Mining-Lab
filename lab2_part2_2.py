import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from pandas.api.types import is_numeric_dtype

# lab 2 part 2.2

missing_values = [" ?"]
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None, na_values = missing_values)
data.columns=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation', 'relationship', 'race', 'sex','capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']

# Histogram "before" for occupation
data['occupation'].hist(bins=14, figsize=(12,8))
plt.xticks(rotation='vertical')
plt.show()

# data['income'][32560]
df1 = data.loc[data['income'] == ' <=50K']
df2 = data.loc[data['income'] == ' >50K']

#finiding averages for the df1 (<=50K)
for col in df1.columns:
#     print(col)
    if is_numeric_dtype(df1[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % df1[col].mean())
        print('\t Standard deviation = %.2f' % df1[col].std())
        print('\t Minimum = %.2f' % df1[col].min())
        print('\t Maximum = %.2f' % df1[col].max())

print()

discrete_attri=['workclass', 'education', 'martial_status', 'occupation', 'race', 'relationship', 'sex', 'native_country','income']

for i in range(0,9):
    print(df1[discrete_attri[i]].value_counts())
    print()

#finiding averages for the df2 (>50K)
for col in df2.columns:
#     print(col)
    if is_numeric_dtype(df2[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % df2[col].mean())
        print('\t Standard deviation = %.2f' % df2[col].std())
        print('\t Minimum = %.2f' % df2[col].min())
        print('\t Maximum = %.2f' % df2[col].max())

print()

discrete_attri=['workclass', 'education', 'martial_status', 'occupation', 'race', 'relationship', 'sex', 'native_country','income']

for i in range(0,9):
    print(df2[discrete_attri[i]].value_counts())
    print()

# Replacing missing values for <=50K
data['age'] = np.where((data.income == ' <=50K'),data['age'].fillna(36.78),data.age)
data['workclass'] = np.where((data.income == ' <=50K'),data['workclass'].fillna('Private'),data.workclass)
data['fnlwgt'] = np.where((data.income == ' <=50K'),data['fnlwgt'].fillna(190340.87),data.fnlwgt)
data['education'] = np.where((data.income == ' <=50K'),data['education'].fillna('HS-grad'),data.education)
data['education_num'] = np.where((data.income == ' <=50K'),data['education_num'].fillna(9.60),data.education_num)
data['martial_status'] = np.where((data.income == ' <=50K'),data['martial_status'].fillna('Never-married'),data.martial_status)
data['occupation'] = np.where((data.income == ' <=50K'),data['occupation'].fillna('Adm-clerical'),data.occupation)
data['relationship'] = np.where((data.income == ' <=50K'),data['relationship'].fillna('Not-in-family'),data.relationship)
data['race'] = np.where((data.income == ' <=50K'),data['race'].fillna('White'),data.race)
data['sex'] = np.where((data.income == ' <=50K'),data['sex'].fillna('Male'),data.sex)
data['capital_gain'] = np.where((data.income == ' <=50K'),data['capital_gain'].fillna(148.75),data.capital_gain)
data['capital_loss'] = np.where((data.income == ' <=50K'),data['capital_loss'].fillna(53.14),data.capital_loss)
data['hours_per_week'] = np.where((data.income == ' <=50K'),data['hours_per_week'].fillna(38.84),data.hours_per_week)
data['native_country'] = np.where((data.income == ' <=50K'),data['native_country'].fillna('United-States'),data.native_country)
# data['income'] = np.where((data.income == ' <=50K'),data['income'].fillna(3678),data.income)

# Replacing missing values for >50K
data['age'] = np.where((data.income == ' >50K'),data['age'].fillna(44.25),data.age)
data['workclass'] = np.where((data.income == ' >50K'),data['workclass'].fillna('Private'),data.workclass)
data['fnlwgt'] = np.where((data.income == ' >50K'),data['fnlwgt'].fillna(188005.00),data.fnlwgt)
data['education'] = np.where((data.income == ' >50K'),data['education'].fillna('Bachelors'),data.education)
data['education_num'] = np.where((data.income == ' >50K'),data['education_num'].fillna(11.61),data.education_num)
data['martial_status'] = np.where((data.income == ' >50K'),data['martial_status'].fillna('Married-civ-spouse'),data.martial_status)
data['occupation'] = np.where((data.income == ' >50K'),data['occupation'].fillna('Exec-managerial'),data.occupation)
data['relationship'] = np.where((data.income == ' >50K'),data['relationship'].fillna('Husband'),data.relationship)
data['race'] = np.where((data.income == ' >50K'),data['race'].fillna('White'),data.race)
data['sex'] = np.where((data.income == ' >50K'),data['sex'].fillna('Male'),data.sex)
data['capital_gain'] = np.where((data.income == ' >50K'),data['capital_gain'].fillna(4006.14),data.capital_gain)
data['capital_loss'] = np.where((data.income == ' >50K'),data['capital_loss'].fillna(195.00),data.capital_loss)
data['hours_per_week'] = np.where((data.income == ' >50K'),data['hours_per_week'].fillna(45.47),data.hours_per_week)
data['native_country'] = np.where((data.income == ' >50K'),data['native_country'].fillna('United-States'),data.native_country)
# data['income'] = np.where((data.income == ' >50K'),data['income'].fillna(3678),data.income)

# Histogram "after" for occupation
data['occupation'].hist(bins=14, figsize=(12,8))
plt.xticks(rotation='vertical')
plt.show()
