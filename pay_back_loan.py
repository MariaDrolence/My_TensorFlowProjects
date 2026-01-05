from sys import float_info
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
#mort_acc - 37795#pub_rec_bankruptcies - 535

#plot the loan_status column
plt.figure(figsize=(14,12))
sns.countplot(x='loan_status', data=df_loan)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_status_countplot.png')
plt.close()

#check for correlation between the numerical features
correlation_matrix=df_loan.corr(numeric_only=True)
print(correlation_matrix)
print(correlation_matrix['loan_amnt'].sort_values(ascending=False))
#plot the correlation matrix as a heatmap
plt.figure(figsize=(12,10))
#add annot=True to show the correlation values on the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_correlation_heatmap.png')
plt.close()

#noticed that the features with highest correlation to loan status are:
#installement and loan amount
#we need to check why we have close correlation with the loam ampunt coulmn could be because of the nature of the data is the same as loan status?
#from review of the column descriptions:
#loan_amnt: The listed amount of the loan applied for by the borrower.
#loan_status: Current status of the loan (e.g., Fully Paid, Charged Off, etc.)

#scatter plot between loan amount and installment rate
plt.figure(figsize=(12,10))
sns.scatterplot(x='installment', y='loan_amnt', data=df_loan)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_amount_vs_installment.png')
plt.close()

'''after attempting to do a scatter i noticed that it is not needed here since the prediction is a classification problem 
and not a continous regression problem like in the other files. Hence no need to scale the data or build a neural network model for regression.'''


#it is possible that higher loan amounts are more likely to be charged off or defaulted on
#let's visualize the relationship between loan amount and loan status using a box plot
plt.figure(figsize=(12,10))
sns.boxplot(x='loan_status', y='loan_amnt', data=df_loan)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_status_vs_loan_amount_boxplot.png')
plt.close()
#from the box plot they are slightly the same but we can see that the charged off loans have a higher median loan amount compared to fully paid loans

#histogram of the loan amount column
plt.figure(figsize=(10,6))
sns.histplot(df_loan['loan_amnt'], bins=40, kde=True)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_amount_distribution.png')
plt.close()

#calculate the summary of statistics for the loam amount grouped by loan status
loan_status_summary=df_loan.groupby('loan_status')['loan_amnt'].describe()
print(loan_status_summary)

grade_order=sorted(df_loan['grade'].unique())   
#visualize the loan status by grade
plt.figure(figsize=(14,12))
sns.countplot(x='grade', data=df_loan, hue='loan_status', order=grade_order)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_status_by_grade.png')
plt.close()



#visualize the loan status by sub_grade
plt.figure(figsize=(16,14))
sns.countplot(x='sub_grade', data=df_loan, hue='loan_status')
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_status_by_sub_grade.png')
plt.close()
#to ensure that the sub_grades are in order we can create a custom order list by picking the unique values from the sub_grade column
sub_grade_order=sorted(df_loan['sub_grade'].unique())

plt.figure(figsize=(16,14))
#then use the order parameter in the countplot to set the order of the sub_grades
sns.countplot(x='sub_grade', data=df_loan, hue='loan_status', order=sub_grade_order)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_status_by_sub_grade_ordered.png')
plt.close()

#if we want to zoom in on a specific grade for more detailed analysis, we can filter the dataframe for a specific grade, e.g., 'B' such as F and G
grade_F_G=df_loan[(df_loan['grade']=='F') | (df_loan['grade']=='G')]
plt.figure(figsize=(16,14))
sub_grade_order=sorted(grade_F_G['sub_grade'].unique())
sns.countplot(x='sub_grade', data=grade_F_G, hue='loan_status', order=sub_grade_order)
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_status_by_sub_grade_F_G.png')
plt.close()

#change the loan_status column to numerical values for easier analysis
#Fully Paid-0
#Charged Off-1

df_loan['loan_repaid']=df_loan['loan_status'].map({'Fully Paid':0, 'Charged Off':1})

#proceed with feature engineering and data preprocessing
plt.figure(figsize=(12,10))
df_loan.corr(numeric_only=True)['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/loan_repaid_correlation.png')
plt.close() 


#now we need to proceed to data preprocessing and feature engineering before building a classification model.