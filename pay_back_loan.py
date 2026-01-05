from sys import float_info
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_loan=pd.read_csv('C:/Users/Maria/Documents/GitHub/TensorflowProjects/TensorFlow_FILES/TensorFlow_FILES/DATA/lending_club_loan_two.csv')

#print(df_loan.head())



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


#we can as well check what percentage of the total data is missing for each column
missing_percentage=df_loan.isnull().sum()/len(df_loan)*100
print(missing_percentage)

#lets first examine the emp_title column and emp_length column so as to decide how to handle the missing values
#how many unique job titles are there in the emp_title column
print(df_loan['emp_title'].nunique())
print(df_loan['emp_title'].value_counts().head(20))
#since there are so many unique job titles it will be difficult to convert them into numerical values
#hence we will drop the emp_title column
df_loan=df_loan.drop('emp_title', axis=1)

#handling the emp_length column
#unique values in the emp_length column
#print(df_loan['emp_length'].dropna().unique())
#we can sort the unique values to see them better

sorted(df_loan['emp_length'].dropna().unique())

#a list of the sorted unique values in the emp_length column
emp_length_ordered=['<1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
#plot the emp_length column to visualize the counts of each employment length category
plt.figure(figsize=(14,12))
sns.countplot(x='emp_length', data=df_loan, order=emp_length_ordered, hue='loan_status')
plt.savefig('C:/Users/Maria/Documents/GitHub/TensorflowProjects/emp_length_countplot.png')
plt.close()

#now we can get the percentage of charged off loans for each emp_length category
#this help us know what percentage of people with a certain employment length are likely to default on their loans
emp_length_charged_off=df_loan[df_loan['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']

emp_length_fully_paid=df_loan[df_loan['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']

emp_length_charged_off_percentage=(emp_length_charged_off/(emp_length_charged_off+emp_length_fully_paid))*100

#noticed the percentage of charged off loans for each employment length category has no big difference over the years
#therefore we will drop the emp_length column since it does not provide much useful information
df_loan=df_loan.drop('emp_length', axis=1)

print(df_loan.isnull().sum())
#percentage of people who charged off over those who fully paid their loans based on employment length

#scince the title column is similar to the purpose column we can drop it
df_loan=df_loan.drop('title', axis=1)

#now let's look at the mort_acc column
print(df_loan['mort_acc'].value_counts())
#we need to check which columns are highly correlated with the mort_acc column
print(correlation_matrix['mort_acc'].sort_values(ascending=False))

#since the mort_acc column has some correlation with the total_acc column we can use that to fill in the missing values
#now we need to fill in the missing values in the mort_acc column based on the average mort_acc for each total_acc value
total_acc_avg_mort_acc = df_loan.groupby('total_acc')['mort_acc'].mean()

print(total_acc_avg_mort_acc)

#function to fill in the missing mort_acc values
def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg_mort_acc[total_acc]
    else:
        return mort_acc 
#apply the function to the mort_acc column
df_loan['mort_acc']=df_loan.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
print(df_loan.isnull().sum())

#now we need to proceed to data preprocessing and feature engineering before building a classification model.
