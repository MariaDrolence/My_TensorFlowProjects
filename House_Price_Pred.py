import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

#load the dataset
df_house=pd.read_csv('C:/Users/Maria/Documents/GitHub/TensorflowProjects/TensorFlow_FILES/TensorFlow_FILES/DATA/kc_house_data.csv')

#view the first few rows of the dataset
print(df_house.head())

#view the summary information about the dataset
print(df_house.info())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
