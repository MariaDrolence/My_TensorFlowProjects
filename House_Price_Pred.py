import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

#load the dataset
df_house=pd.read_csv('C:/Users/Maria/Documents/GitHub/TensorflowProjects/TensorFlow_FILES/TensorFlow_FILES/DATA/kc_house_data.csv')

#view the first few rows of the dataset
#print(df_house.head())

#view the summary information about the dataset
#print(df_house.info())
#from the infor we don't have any missing values

#this can also be checked using isnull().sum()
#print(df_house.isnull().sum())

#view the statistical summary of the dataset-such as mean, std, min, max, and percentiles
#print(df_house.describe())

#visualize the distribution of the target variable 'price'
plt.figure(figsize=(10,6))
sns.displot(df_house['price'])
#plt.title('Distribution of House Prices')
#plt.xlabel('Price')
#plt.ylabel('Frequency')
#plt.show()
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/house_price_distribution.png')

sns.countplot(x='bedrooms', data=df_house)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/bedrooms_count.png')
#plt.show()


#adding a boxplot to see the spread of prices for different number of bedrooms
plt.figure(figsize=(10,6))
sns.boxplot(x='bedrooms', y='price', data=df_house)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/bedrooms_vs_price.png')
#plt.show()

#finding the correlation between different features in the dataset to the target variable 'price'
correlation_matrix=df_house.corr(numeric_only=True)
print(correlation_matrix['price'].sort_values(ascending=False))

sns.scatterplot(x='sqft_living', y='price', data=df_house)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/sqft_living_vs_price.png')
#plt.show()

#scatter plot for latitude and longitude
plt.figure(figsize=(10,6))
sns.scatterplot(x='long', y='lat', data=df_house, hue='price', palette='viridis')
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/location_price_scatter.png')


#cleaning and preprocessing would be done in the model building file
df_house=df_house.drop(['id'], axis=1)

#convert the 'date' column to datetime format
df_house['date']=pd.to_datetime(df_house['date'])

#extracting year and month from the date column
df_house['year']=df_house['date'].dt.year
df_house['month']=df_house['date'].dt.month

#drop the original date column after extracting useful information
df_house=df_house.drop('date', axis=1)

#a plot to see average price per month
df_house.groupby('month').mean()['price'].plot()

df_house.groupby('month').mean()['price'].plot()
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/average_price_per_month.png')

#print(df_house.head())

#check the number of unique zipcodes
print(df_house['zipcode'].nunique())
#check the different zipcodes and their counts
print(df_house['zipcode'].value_counts())

#drop the zipcode column since it has too many unique values
df_house=df_house.drop('zipcode', axis=1)
#according to the dataset 0 year means the house was never renovated
#we can create a new column to indicate whether a house was renovated
print(df_house['yr_renovated'].value_counts())

#from the dataset we notice that as the years increase the number of renovation increase so having 0 means many house aerlier on were not renovated
#if the year is 0 we will assign 0 else 1
df_house['renovated']=df_house['yr_renovated'].apply(lambda x: 0 if x==0 else 1)
#drop the previous yr_renovated column
df_house=df_house.drop('yr_renovated', axis=1)

print(df_house['sqft_basement'].value_counts())

#creating a new column to indicate whether a house has a basement or not
df_house['has_basement']=df_house['sqft_basement'].apply(lambda x: 0 if x==0 else 1)
#drop the previous sqft_basement column
df_house=df_house.drop('sqft_basement', axis=1)

#original features and target variable
#DROP the price column from the features and store it in y using .values to convert it into an array
X=df_house.drop('price', axis=1).values
y=df_house['price'].values

#importing necessary libraries for model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler

#create an instance of the minmaxscaler
scaler= MinMaxScaler()
#fit the scaler to the training data and transform both training and testing data-Compute the minimum and maximum to be used for later scaling
scaler.fit(X_train)

#transform the training and testing data
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
