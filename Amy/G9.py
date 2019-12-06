#!/usr/bin/env python
# coding: utf-8

# ### Core Analysis
# *Ming-Chen Lu*

# In[49]:


# Stats 506, Fall 2019
# Group Project - group 9
#
# This script aims to explore the relastionship between drinking habits and 
# general health condition for adults over 21 years old in US.
#
# NHANES 2005-2006 Questionnaire Data: Alcohol Use, Current Health Status
# Source: https://wwwn.cdc.gov/nchs/nhanes/Search/DataPage.aspx?Component=Questionnaire&CycleBeginYear=2005
# NHANES 2005-2006 Demographics Data
# Source: https://wwwn.cdc.gov/nchs/nhanes/Search/DataPage.aspx?Component=Demographics&CycleBeginYear=2005
#
# Author: Ming-Chen Lu
# Updated: November 29, 2019
#80: ---------------------------------------------------------------------------

# Set up: ----------------------------------------------------------------------
import xport
import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from scipy.stats import t
from statsmodels.formula.api import ols

# Read in the data: ------------------------------------------------------------
## 'rb' mode - opens the file in binary format for reading
with open('HSQ_D.XPT', 'rb') as f:
    df_health = xport.to_dataframe(f)

with open('ALQ_D.XPT', 'rb') as f:
    df_alcohol = xport.to_dataframe(f)

with open('DEMO_D.XPT', 'rb') as f:
    df_demo = xport.to_dataframe(f)
    
# Data preparation: ------------------------------------------------------------
# Extract key columns
df_health = df_health.loc[df_health['HSD010'] <= 3, ['SEQN','HSD010']]
df_alcohol = df_alcohol.loc[df_alcohol['ALQ120Q'] <= 365, ['SEQN','ALQ120Q']]
df_demo = df_demo.loc[(df_demo.RIDAGEYR >= 21) & (df_demo.DMDEDUC2 <= 5), 
                      ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'INDFMPIR', 'DMDEDUC2']]

# Merge key columns into one data frame
df = pd.merge(df_alcohol, df_health, on = 'SEQN')
df = pd.merge(df, df_demo, on = 'SEQN')

# Drop missing values
#df.isnull().sum()
df = df.dropna(axis = 0)

# Rename columns
df = df.rename(columns = {"SEQN": "id", "ALQ120Q": "alcohol", "HSD010": "health",
                          "RIAGENDR": "sex", "RIDAGEYR": "age", "INDFMPIR": "pir",
                          "DMDEDUC2": "edu"})

# Normalize alcohol, age, and poverty income ratio(pir)
df.alcohol = (df.alcohol - np.mean(df.alcohol)) / np.std(df.alcohol)
df.age = (df.age - np.mean(df.age)) / np.std(df.age)
df.pir = (df.pir - np.mean(df.pir)) / np.std(df.pir)
# Factorize health and education
df.sex = df.sex.astype('category')
df.edu = df.edu.astype('category')
df.health = pd.factorize(df.health)[0]

# Initial linear regression model: ---------------------------------------------
lmod = ols('alcohol ~ C(health) + C(sex) + age + pir + C(edu)', data = df).fit()
print('')
lmod.summary()

# plot residual errors
fitted = lmod.fittedvalues
res = lmod.resid
plt.style.use('ggplot')
plt.scatter(fitted, res, color = "green", s = 10)
plt.hlines(y = 0, xmin = -0.16, xmax = 0.15, linewidth = 1, color = "red")
plt.title("Residuals vs Fitted")
plt.show()

# Function to resample residuals with replacement and fit new model: -----------
def boot_res(df, fitted, res):
    # input: 
    #   df - the original data for fitting initial model
    #   fitted - a array of fitted values from the initial model
    #   res  - a array of residuals from the initial model
    # output: 
    #   n_lmod.params - the coefficients of new model
    #   n_ss - sigma square for the additional analysis
    
    # sampling residauls with replacement
    b_res = np.random.choice(res, size = len(res), replace = True)
    n_ss = sum(b_res**2) / (df.shape[0] - df.shape[1] - 1)
    
    # adding the resampled residuals back to the fitted values
    new_y = fitted + b_res
    
    # combine old predictors values with new responses
    X = df.iloc[:,2:]
    n_df = pd.concat([new_y, X], axis = 1)
    n_df.rename({0:'alcohol'}, axis = 1, inplace = True) 
    
    # fit new model
    n_lmod = ols('alcohol ~ C(health) + C(sex) + age + pir + C(edu)', data = n_df).fit()
    return(n_lmod.params, n_ss)

# Test the function
#boot_res(df, fitted, res)

# Bootstrapping residuals 1000 times: ------------------------------------------
random.seed(506)
B = 1000
b = [boot_res(df, fitted, res) for i in range(B)]
b_coef = [lis[0] for lis in b]

# convert list to dataframe
b_df = pd.DataFrame(np.row_stack(np.array(b_coef)), 
                    columns = ['Intercept', 'health.1', 'health.2', 'sex.2', 'edu.2',
                               'edu.3', 'edu.4', 'edu.5', 'age', 'pir'])

# Compute SE for 1000 times bootstrap
b_se = b_df.std(axis = 0)
#print("Standard Error for each coefficient:", b_se)

# Plot the distribution of bootstrapping coefficients for "health" variable
#bh1 = sns.distplot(b_df.iloc[:,1])
#bh1.set_title('Coefficients Distribution of Health "very good"')

# Compute t-statistic
tval = np.array(lmod.params)[:] / np.array(b_se)

# Compute p-value
pval = t.sf(np.abs(tval), 1)

# Combine result into a dataframe
col = ["Estimate", "SE", "tStats", "pValue"]
rows = lmod.params.index.values
data = np.array([lmod.params, b_se, tval, pval])
data = np.transpose(data)
tbl = pd.DataFrame(data=data, columns=col, index=rows)
print(tbl)


# ## Additional Analysis

# In[57]:


from patsy import dmatrix

# Extract se from 1000 times bootstrapping
n_ss = [lis[1] for lis in b]

# Take the minimum SE after 1000 times bootstrapping
n_ss = [lis[1] for lis in b]
n_ss = min(n_ss)

# Extract the design matrix of predictors
design_matrix = dmatrix('C(health) + C(sex) + age + pir + C(edu)', df)

# Compute the standard errors of coefficients using bootstrapping residuals
XX = np.linalg.inv(np.dot(np.transpose(predictors), predictors))
n_se = np.sqrt(np.diag(n_ss*XX))

# Compute t-statistic
aa_tval = np.array(lmod.params[:]) / n_se
aa_tval

# Compute p-value
aa_pval = t.sf(np.abs(aa_tval), 1)

# Combine result into a dataframe
aa_data = {'Estimate': np.array(lmod.params[:]), 'SE': n_se, 
           'tStats': aa_tval, 'pValue': aa_pval}
rows = lmod.params.index.values
aa_tbl = pd.DataFrame(data=aa_data, index=rows)
print(aa_tbl)
