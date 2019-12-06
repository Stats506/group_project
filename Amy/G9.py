#!/usr/bin/env python
# coding: utf-8

# ### Core Analysis
# *Ming-Chen Lu*

# In[1]:


# Stats 506, Fall 2019
# Group Project - group 9
#{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Core Analysis\n",
    "*Ming-Chen Lu*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Stats 506, Fall 2019\n",
    "# Group Project - group 9\n",
    "#\n",
    "# This script aims to explore the relastionship between drinking habits and \n",
    "# general health condition for adults over 21 years old in US.\n",
    "#\n",
    "# NHANES 2005-2006 Questionnaire Data: Alcohol Use, Current Health Status\n",
    "# Source: https://wwwn.cdc.gov/nchs/nhanes/Search/DataPage.aspx?Component=Questionnaire&CycleBeginYear=2005\n",
    "# NHANES 2005-2006 Demographics Data\n",
    "# Source: https://wwwn.cdc.gov/nchs/nhanes/Search/DataPage.aspx?Component=Demographics&CycleBeginYear=2005\n",
    "#\n",
    "# Author: Ming-Chen Lu\n",
    "# Updated: November 29, 2019\n",
    "#80: ---------------------------------------------------------------------------\n",
    "\n",
    "# Set up: ----------------------------------------------------------------------\n",
    "import xport\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import math\n",
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.stats import t\n",
    "from statsmodels.formula.api import ols\n",
    "\n",
    "# Read in the data: ------------------------------------------------------------\n",
    "## 'rb' mode - opens the file in binary format for reading\n",
    "with open('HSQ_D.XPT', 'rb') as f:\n",
    "    df_health = xport.to_dataframe(f)\n",
    "\n",
    "with open('ALQ_D.XPT', 'rb') as f:\n",
    "    df_alcohol = xport.to_dataframe(f)\n",
    "\n",
    "with open('DEMO_D.XPT', 'rb') as f:\n",
    "    df_demo = xport.to_dataframe(f)\n",
    "    \n",
    "# Data preparation: ------------------------------------------------------------\n",
    "# Extract key columns\n",
    "df_health = df_health.loc[df_health['HSD010'] <= 3, ['SEQN','HSD010']]\n",
    "df_alcohol = df_alcohol.loc[df_alcohol['ALQ120Q'] <= 365, ['SEQN','ALQ120Q']]\n",
    "df_demo = df_demo.loc[(df_demo.RIDAGEYR >= 21) & (df_demo.DMDEDUC2 <= 5), \n",
    "                      ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'INDFMPIR', 'DMDEDUC2']]\n",
    "\n",
    "# Merge key columns into one data frame\n",
    "df = pd.merge(df_alcohol, df_health, on = 'SEQN')\n",
    "df = pd.merge(df, df_demo, on = 'SEQN')\n",
    "\n",
    "# Drop missing values\n",
    "#df.isnull().sum()\n",
    "df = df.dropna(axis = 0)\n",
    "\n",
    "# Rename columns\n",
    "df = df.rename(columns = {\"SEQN\": \"id\", \"ALQ120Q\": \"alcohol\", \"HSD010\": \"health\",\n",
    "                          \"RIAGENDR\": \"sex\", \"RIDAGEYR\": \"age\", \"INDFMPIR\": \"pir\",\n",
    "                          \"DMDEDUC2\": \"edu\"})\n",
    "\n",
    "# Normalize alcohol, age, and poverty income ratio(pir)\n",
    "df.alcohol = (df.alcohol - np.mean(df.alcohol)) / np.std(df.alcohol)\n",
    "df.age = (df.age - np.mean(df.age)) / np.std(df.age)\n",
    "df.pir = (df.pir - np.mean(df.pir)) / np.std(df.pir)\n",
    "# Factorize health and education\n",
    "df.sex = df.sex.astype('category')\n",
    "df.edu = df.edu.astype('category')\n",
    "df.health = pd.factorize(df.health)[0]\n",
    "\n",
    "# Initial linear regression model: ---------------------------------------------\n",
    "lmod = ols('alcohol ~ C(health) + C(sex) + age + pir + C(edu)', data = df).fit()\n",
    "print('')\n",
    "lmod.summary()\n",
    "\n",
    "# plot residual errors\n",
    "fitted = lmod.fittedvalues\n",
    "res = lmod.resid\n",
    "plt.style.use('ggplot')\n",
    "plt.scatter(fitted, res, color = \"green\", s = 10)\n",
    "plt.hlines(y = 0, xmin = -0.16, xmax = 0.15, linewidth = 1, color = \"red\")\n",
    "plt.title(\"Residuals vs Fitted\")\n",
    "plt.show()\n",
    "\n",
    "# Function to resample residuals with replacement and fit new model: -----------\n",
    "def boot_res(df, fitted, res):\n",
    "    # input: \n",
    "    #   df - the original data for fitting initial model\n",
    "    #   fitted - a array of fitted values from the initial model\n",
    "    #   res  - a array of residuals from the initial model\n",
    "    # output: \n",
    "    #   n_lmod.params - the coefficients of new model\n",
    "    #   n_se - standard error for the additional analysis\n",
    "    \n",
    "    # sampling residauls with replacement\n",
    "    b_res = np.random.choice(res, size = len(res), replace = True)\n",
    "    n_se = math.sqrt( sum(b_res**2) / (df.shape[0] - df.shape[1] - 1) )\n",
    "    \n",
    "    # adding the resampled residuals back to the fitted values\n",
    "    new_y = fitted + b_res\n",
    "    \n",
    "    # combine old predictors values with new responses\n",
    "    X = df.iloc[:,2:]\n",
    "    n_df = pd.concat([new_y, X], axis = 1)\n",
    "    n_df.rename({0:'alcohol'}, axis = 1, inplace = True) \n",
    "    \n",
    "    # fit new model\n",
    "    n_lmod = ols('alcohol ~ C(health) + C(sex) + age + pir + C(edu)', data = n_df).fit()\n",
    "    return(n_lmod.params, n_se)\n",
    "\n",
    "# Test the function\n",
    "#boot_res(df, fitted, res)\n",
    "\n",
    "# Bootstrapping residuals 1000 times: ------------------------------------------\n",
    "random.seed(506)\n",
    "B = 1000\n",
    "b = [boot_res(df, fitted, res) for i in range(B)]\n",
    "b_coef = [lis[0] for lis in b]\n",
    "\n",
    "# convert list to dataframe\n",
    "b_df = pd.DataFrame(np.row_stack(np.array(b_coef)), \n",
    "                    columns = ['Intercept', 'health.1', 'health.2', 'sex.2', 'edu.2',\n",
    "                               'edu.3', 'edu.4', 'edu.5', 'age', 'pir'])\n",
    "\n",
    "# Compute SE for 1000 times bootstrap\n",
    "b_se = b_df.std(axis = 0)\n",
    "#print(\"Standard Error for each coefficient:\", b_se)\n",
    "\n",
    "# Plot the distribution of bootstrapping coefficients for \"health\" variable\n",
    "#bh1 = sns.distplot(b_df.iloc[:,1])\n",
    "#bh1.set_title('Coefficients Distribution of Health \"very good\"')\n",
    "\n",
    "# Compute t-statistic\n",
    "tval = np.array(lmod.params)[:] / np.array(b_se)\n",
    "\n",
    "# Compute p-value\n",
    "pval = t.sf(np.abs(tval), 1)\n",
    "\n",
    "# Combine result into a dataframe\n",
    "col = [\"Estimate\", \"SE\", \"tStats\", \"pValue\"]\n",
    "rows = lmod.params.index.values\n",
    "data = np.array([lmod.params, b_se, tval, pval])\n",
    "data = np.transpose(data)\n",
    "tbl = pd.DataFrame(data=data, columns=col, index=rows)\n",
    "tbl"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

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
import math
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


# The residual versus fitted plot presents the non-normality issue. In order to trust our model more and make the standard errors more robust, we resample the residuals to the estimate and adding them back to the fitted values.

# In[109]:


# Function to resample residuals with replacement and fit new model: -----------
def boot_res(df, fitted, res):
    # input: 
    #   df - the original data for fitting initial model
    #   fitted - a array of fitted values from the initial model
    #   res  - a array of residuals from the initial model
    # output: 
    #   n_lmod.params - the coefficients of new model
    #   n_se - standard error for the additional analysis
    
    # sampling residauls with replacement
    b_res = np.random.choice(res, size = len(res), replace = True)
    n_se = math.sqrt( sum(b_res**2) / (df.shape[0] - df.shape[1] - 1) )
    
    # adding the resampled residuals back to the fitted values
    new_y = fitted + b_res
    
    # combine old predictors values with new responses
    X = df.iloc[:,2:]
    n_df = pd.concat([new_y, X], axis = 1)
    n_df.rename({0:'alcohol'}, axis = 1, inplace = True) 
    
    # fit new model
    n_lmod = ols('alcohol ~ C(health) + C(sex) + age + pir + C(edu)', data = n_df).fit()
    return(n_lmod.params, n_se)

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
tbl

