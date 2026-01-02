import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_cancer=pd.read_csv('C:/Users/Maria/Documents/GitHub/TensorflowProjects/TensorFlow_FILES/TensorFlow_FILES/DATA/cancer_classification.csv')   

#exploratory data analysis
print(df_cancer.head())

print(df_cancer.info())

print(df_cancer.describe())

print(df_cancer.isnull().sum())

# Visualizing the distribution of the target variable

sns.displot(df_cancer['benign_0__mal_1'])
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/cancer_target_distribution.png')


plt.figure(figsize=(10,6))
sns.countplot(x='benign_0__mal_1', data=df_cancer)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/cancer_target_count.png')


# Correlation matrix to see relationships between features
correlation_matrix=df_cancer.corr(numeric_only=True)
print(correlation_matrix['benign_0__mal_1'].sort_values(ascending=False))
#plot the corrlation matrix as a barplot
plt.figure(figsize=(12,10))
correlation_matrix['benign_0__mal_1'].sort_values(ascending=False).plot(kind='bar')
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/cancer_feature_correlation.png')

#plot a heatmap of the correlation matrix
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/cancer_correlation_heatmap.png')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X=df_cancer.drop('benign_0__mal_1', axis=1).values
y=df_cancer['benign_0__mal_1'].values

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=101)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))

# Output layer-for binary classification the activation function is sigmoid
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




