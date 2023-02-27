import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from pandas.api.types import is_numeric_dtype

# lab 2 part 2.1
missing_values = [" ?"]
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None, na_values = missing_values)
data.columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 'race', 'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']

# Histogram "before" for occupation
data['occupation'].hist(bins=14, figsize=(12,8))
plt.xticks(rotation='vertical')
plt.show()

data['age'] = data['age'].fillna(38.58)
data['workclass'] = data['workclass'].fillna('Private')
data['fnlwgt'] = data['fnlwgt'].fillna(189778.37)
data['education'] = data['education'].fillna('HS-grad')
data['education-num'] = data['education-num'].fillna(10.08)
data['martial-status'] = data['martial-status'].fillna('Married-civ-spouse')
data['occupation'] = data['occupation'].fillna('Prof-specialty')
data['relationship'] = data['relationship'].fillna('Husband')
data['race'] = data['race'].fillna('White')
data['sex'] = data['sex'].fillna('Male')
data['capital-gain'] = data['capital-gain'].fillna(1077.65)
data['capital-loss'] = data['capital-loss'].fillna(87.30)
data['hours-per-week'] = data['hours-per-week'].fillna(40.44)
data['native-country'] = data['native-country'].fillna('UnitedStates')
data['income'] = data['income'].fillna(' <=50K')

# Histogram "after" for occupation
data['occupation'].hist(bins=14, figsize=(12,8))
plt.xticks(rotation='vertical')
plt.show()
