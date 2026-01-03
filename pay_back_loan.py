import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_loan=pd.read_csv('C:/Users/Maria/Documents/GitHub/TensorflowProjects/TensorFlow_FILES/TensorFlow_FILES/DATA/lending_club_loan_two.csv')

#print(df_loan.head())

print(df_loan.info())
#from the info we can see that there are some null values in the columns

#checking the null values in each column
print(df_loan.isnull().sum())
#noticed that the columns with missing value are:
df_loan.isnull().sum()

#noticed that the columns with missing value are:
#emp_title - 22927
#emp_length - 18301
#title - 1756
#mort_acc - 37795
#pub_rec_bankruptcies - 535
#revol_util - 276