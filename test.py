
import pandas as pd

import numpy as np

import seaborn as sns
import matplotlib .pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


df= pd.read_csv("C:/Users/Maria/Documents/GitHub/TensorflowProjects/TensorFlow_FILES/TensorFlow_FILES/DATA/fake_reg.csv")


#plt.figure(figsize=(10,6))
#sns.pairplot(df)
#plt.show()

from sklearn.model_selection import train_test_split

#store the features in form of an array since we are going to use tensorflow

X=df[['feature1','feature2']].values
y=df['price'].values

#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.preprocessing import MinMaxScaler

#create an instance of the minmaxscaler
scaler= MinMaxScaler()
#fit the scaler to the training data and transform both training and testing data-Compute the minimum and maximum to be used for later scaling
scaler.fit(X_train)
#transform the training and testing data
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

#print(X_train_scaled.min(), X_train_scaled.max())
#Build the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#create the empty sequential model
model=Sequential()
#add the layers
model.add(Dense(4, activation='relu', input_shape=(2,)))
model.add(Dense(4, activation='relu', input_shape=(2,)))
model.add(Dense(4, activation='relu', input_shape=(2,)))

#final output layer
model.add(Dense(1))#no activation function since its a regression problem

#compile the model the rmsprop- optimization algorithm is used for trainign regression models
model.compile(optimizer='rmsprop', loss='mse')

#train the model
#epochs-number of iterations the model will run through the entire x and y data
#verbose- controls the output that is printed to the console during training
history=model.fit(X_train_scaled, y_train, epochs=250, verbose=1)

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#Evaluate the model on the test data
test_loss=model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test Loss:", test_loss)

#make predictions
y_pred=model.predict(X_test_scaled)
print(y_pred)

test_predictions=pd.Series(y_pred.flatten(), name='Predicted Values')

#print(test_predictions)
test_predictions=pd.DataFrame(test_predictions)
test_predictions['Actual Values']=y_test
print(test_predictions)

sns.scatterplot(x='Actual Values', y='Predicted Values', data=test_predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

#calculate the mean absolute error, mean squared error and root mean squared error
from sklearn.metrics import mean_absolute_error, mean_squared_error

#calculate the mean absolute error, mean absolute error 
mae=mean_absolute_error(y_test, y_pred)

#calculate the mean absolute error, mean squared error 
mse=mean_squared_error(y_test, y_pred)

#calculate the root mean squared error
rmse=np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

print(df.describe())

#save the model

from tensorflow.keras.models import load_model

model.save('my_model.keras')

#load the model
new_model=load_model('my_model.keras')