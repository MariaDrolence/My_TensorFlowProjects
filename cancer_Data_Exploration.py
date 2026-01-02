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

# Compile the model--for binary classification the loss function is binary_crossentropy--optimizer can be adam.
# note that compiling means configuring the model for training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using the fit method
#model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))

#losses=pd.DataFrame(model.history.history)
#plot the losses
#when the validation loss starts increasing while the training loss is decreasing, it indicates overfitting to the training data.
#plt.figure(figsize=(14,12))
#losses.plot()
#plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/cancer_model_loss_accuracy.png')    


#to solve overfitting, we can use callbacks such as early stopping or dropout layers during training.

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# Train the model using the fit method- with early stopping which stops training when the validation loss does not improve for 25 consecutive epochs
#model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

#plot the losses again to see the effect of early stopping
##plt.figure(figsize=(14,12))
#model_loss=pd.DataFrame(model.history.history)  

#model_loss.plot()
#plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/cancer_model_loss_accuracy_early_stopping.png')

#another way to solve overfitting is to use dropout layers in the model architecture.
from tensorflow.keras.layers import Dropout

model_dropout=Sequential()
model_dropout.add(Dense(30, activation='relu'))
model_dropout.add(Dropout(0.5))  # Dropout layer with 50% dropout rate-which means that 50% of the neurons will be randomly turned off during each training iteration
model_dropout.add(Dense(15, activation='relu'))
model_dropout.add(Dropout(0.5))  # Dropout layer with 50% dropout rate
model_dropout.add(Dense(1, activation='sigmoid'))

model_dropout.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_dropout.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

#plot the losses again to see the effect of dropout layers
#plt.figure(figsize=(14,12))
#model_dropout_loss=pd.DataFrame(model_dropout.history.history)
#model_dropout_loss.plot()
#plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/cancer_model_loss_accuracy_dropout.png')

# predicting on the test data

predictions=model_dropout.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))




