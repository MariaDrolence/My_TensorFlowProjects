import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

#load the dataset
df_house=pd.read_csv('C:/Users/Maria/Documents/GitHub/TensorflowProjects/TensorFlow_FILES/TensorFlow_FILES/DATA/kc_house_data.csv')

#view the first few rows of the dataset
print(df_house.head())

#view the summary information about the dataset
print(df_house.info())
#from the infor we don't have any missing values

#this can also be checked using isnull().sum()
print(df_house.isnull().sum())

#view the statistical summary of the dataset-such as mean, std, min, max, and percentiles
print(df_house.describe())

#visualize the distribution of the target variable 'price'
plt.figure(figsize=(10,6))
sns.displot(df_house['price'])
#plt.title('Distribution of House Prices')
#plt.xlabel('Price')
#plt.ylabel('Frequency')
plt.show()
#plt.savefig('house_price_distribution.png')

sns.countplot(x='bedrooms', data=df_house)
plt.show()

#finding the correlation between different features in the dataset to the target variable 'price'
correlation_matrix=df_house.corr(numeric_only=True)
print(correlation_matrix['price'].sort_values(ascending=False))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
