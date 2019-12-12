{
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
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "import xport\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
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
    "    df_demo = xport.to_dataframe(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                Estimate        SE    tStats    pValue\n",
      "Intercept      -0.097679  0.076945 -1.269469  0.212381\n",
      "C(health)[T.1]  0.049538  0.041860  1.183424  0.223322\n",
      "C(health)[T.2]  0.059684  0.059147  1.009076  0.248562\n",
      "C(sex)[T.2.0]  -0.066729  0.037728 -1.768665  0.163799\n",
      "C(edu)[T.2.0]   0.102236  0.091110  1.122120  0.231703\n",
      "C(edu)[T.3.0]   0.101336  0.083423  1.214728  0.219235\n",
      "C(edu)[T.4.0]   0.141714  0.082418  1.719456  0.167674\n",
      "C(edu)[T.5.0]   0.087347  0.088699  0.984758  0.252444\n",
      "age            -0.005543  0.018961 -0.292329  0.409471\n",
      "pir             0.016709  0.021945  0.761422  0.292853\n"
     ]
    }
   ],
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
    "    #   n_ss - sigma square for the additional analysis\n",
    "    \n",
    "    # sampling residauls with replacement\n",
    "    b_res = np.random.choice(res, size = len(res), replace = True)\n",
    "    n_ss = sum(b_res**2) / (df.shape[0] - df.shape[1] - 1)\n",
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
    "    return(n_lmod.params, n_ss)\n",
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
    "print(tbl)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Additional Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXwAAAEJCAYAAACXCJy4AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAAIABJREFUeJzs3Xl4U1X++PF39jRtum9A2UvZi7LvuIIgCIiOooJ+VRB/MozMoKKggAKD6IiCjiCiDAijjDqoMwwCIrKL7GUtUKCU7nvSJdu9vz8KgdpC25A2aXpez9PnocnJvZ97Ez49Offcz1HIsiwjCIIg+DylpwMQBEEQ6oZI+IIgCA2ESPiCIAgNhEj4giAIDYRI+IIgCA2ESPiCIAgNhEj4giDUmq1bt7J//363b/f06dN89913bt+urxMJvxalpKTQvn17Ro4cyciRIxkxYgSPPvooGzZscLb54IMPWL9+/U238+GHH7Jly5ZKn7v+9W3btiU3N7dGMR49epQ33ngDgISEBKZMmVKj19/MXXfdxZAhQxg5ciSjRo3i/vvvZ/78+UiSREpKCrfffnuV2/jXv/7FmjVrarRfs9nMo48+yv3338+mTZvKPbdkyRJ69+7tfE+u/rz77rs33ea2bdv44IMPAPjpp5+YO3dujWK6mZkzZ3Ls2LEavSY3N5e2bdtW+lxmZiYvvvgiI0aMYMSIETz88MPlPj8jR46ksLDwlmKuDovFwrx585g3b57zsSVLlvDmm2/WeFtPP/10uc/2okWLWLBgAZmZmc7HvvvuOx544AFGjhzJo48+SkJCwq0dgA9SezoAX6fX68v1RC5fvsxTTz2FSqViyJAh/OlPf6pyG7/++iuxsbGVPled19/M2bNnycjIAKBz584sXrz4lrb3e++++y6dO3cGwGq1Mm7cONauXcsdd9xRrdcfOHCANm3a1GifJ0+eJCcnh82bN1f6/LBhw5x/5KorISGBgoICAO6++27uvvvuGr3+Znbv3s0jjzzitu3NnDmTvn378v777wNl7/HYsWNp2bIlrVu3rrOe8TfffMMdd9zB5cuX2b59OwMHDnR5W7t27XL++9y5c6SmpjJx4kRWrVrFtGnTSEpK4p133uHbb78lMjKSX375hT/+8Y9s27bNDUfiO0TCr2NNmjRhypQprFixgiFDhjB9+nTatGnDM888w+LFi9m8eTMajYaQkBD++te/snnzZo4dO8bChQtRqVT89NNP5Ofnc+nSJe644w5ycnKcrwd4//33SUhIQJIkXnzxRe68806+/fZbfvzxR5YtWwbg/H327NksXrwYk8nEq6++yqhRo3jrrbf4z3/+g8lkYs6cOZw6dQqFQsGAAQP485//jFqtpnPnzkycOJFdu3aRmZnJs88+y2OPPVblsWu1Wrp160ZSUlK5hG+z2ViwYAF79uxBpVIRHx/Pq6++yp49e9i6dSu7du1Cr9fz+OOPl9veli1b+PDDD5EkCX9/f1599VUCAgJ47bXXyMjIYOTIkXz11Vfo9fpqvz+bNm3i448/RqFQoFKpePnll9FqtXz55Zc4HA6MRiPNmzd3ns9x48bRsWNHDh8+TG5uLn/4wx/Izs5m3759lJSU8P7779O2bVsOHz7MO++8g9VqJSsri759+zJ//nwWLVpEZmYm06ZNY+HChbRq1Yp58+aRmJiIzWajT58+vPzyy6jVajZt2sSiRYvw8/OjU6dONzyGrKwsSktLkSQJpVJJbGwsH3/8MYGBgUDZN8E9e/YQFBTEwoUL2bp1K0ajkfj4eM6dO8fq1atv+bgkSWLlypV8/PHHpKens2zZMmfCT0pKYty4cWRlZREeHs57771HZGQkP//8M8uWLcNqtZKbm8uoUaN48cUXefXVVwF48skn+eSTT/j0008ZO3Ys999/P8OHD2fSpElotVrmzp1LZGQkAJ06dSI7Oxur1YpWq632++/zZKHWXLp0Sb7tttsqPJ6YmCh36dJFlmVZfuWVV+RPP/1UTk1Nlbt27SpbLBZZlmV5xYoV8ubNm2VZluUnnnhC/t///uds/+STTzq3dfX1sizLcXFx8rJly2RZluXTp0/LPXv2lHNycuRvvvlGnjhxovM11/9+/b/37t0r33///bIsy/LLL78sv/XWW7IkSbLFYpGffvpp57bj4uLk1atXy7IsywkJCXKnTp3k0tLSCsd55513ykePHnX+np6eLt93333yxo0by52bDz74QJ48ebJstVplh8MhT58+XX799dcrHN/1zp49K/ft21dOTk6WZVmWd+/eLffr1082mUzljuP3Fi9eLPfq1Ut+4IEHyv1s375dlmVZvvvuu+VDhw7JsizLO3bskJcsWeJ83Zw5cyqcsyee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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from patsy import dmatrix\n",
    "\n",
    "# Extract sigma^2 from 1000 times bootstrapping\n",
    "n_ss = [lis[1] for lis in b]\n",
    "\n",
    "# distribution plot for sigma^2\n",
    "sns.set()\n",
    "sns.distplot(n_ss)\n",
    "plt.axvline(min(n_ss), linestyle='dashed', linewidth=1.2, color='g', label = 'minimum')\n",
    "plt.axvline(np.median(n_ss), linestyle='dashed', linewidth=1.2, color='k', label = 'median')\n",
    "plt.axvline(0.9992, linestyle='dashed', linewidth=1.2, color='m', label = 'original')\n",
    "plt.axvline(np.mean(n_ss), linestyle='dashed', linewidth=1.2, color='r', label = 'mean')\n",
    "\n",
    "plt.legend()\n",
    "plt.title('Distribution Plot of Estimated Sigma^hat^2')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the distribution plot of ${\\hat{\\sigma}}^{2^*}$, it turns out that the mean of bootstrapping ${\\hat{\\sigma}}^{2^*}$ was is similar to our original ${\\hat{\\sigma}}^{2}$. Hence, we would conclude that the standard errors of the model is robust."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                Estimate        SE    tStats    pValue\n",
      "Intercept      -0.097679  0.040208 -2.429361  0.124298\n",
      "C(health)[T.1]  0.049538  0.021727  2.279988  0.131568\n",
      "C(health)[T.2]  0.059684  0.031991  1.865654  0.156620\n",
      "C(sex)[T.2.0]  -0.066729  0.019867 -3.358825  0.092108\n",
      "C(edu)[T.2.0]   0.102236  0.046970  2.176609  0.137086\n",
      "C(edu)[T.3.0]   0.101336  0.043234  2.343906  0.128361\n",
      "C(edu)[T.4.0]   0.141714  0.043505  3.257426  0.094811\n",
      "C(edu)[T.5.0]   0.087347  0.046729  1.869226  0.156366\n",
      "age            -0.005543  0.010028 -0.552747  0.339270\n",
      "pir             0.016709  0.011412  1.464216  0.190730\n"
     ]
    }
   ],
   "source": [
    "from patsy import dmatrix\n",
    "\n",
    "# Extract sigma^2 from 1000 times bootstrapping\n",
    "n_ss = [lis[1] for lis in b]\n",
    "\n",
    "# Take the minimum sigma^2 after 1000 times bootstrapping\n",
    "min_ss = min(n_ss)\n",
    "\n",
    "# Extract the design matrix of predictors\n",
    "design_matrix = dmatrix('C(health) + C(sex) + age + pir + C(edu)', df)\n",
    "\n",
    "# Compute the standard errors of coefficients using bootstrapping residuals\n",
    "XX = np.linalg.inv(np.dot(np.transpose(design_matrix), design_matrix))\n",
    "min_se = np.sqrt(np.diag(min_ss*XX))\n",
    "\n",
    "# Compute t-statistic\n",
    "aa_tval = np.array(lmod.params[:]) / min_se\n",
    "\n",
    "# Compute p-value\n",
    "aa_pval = t.sf(np.abs(aa_tval), 1)\n",
    "\n",
    "# Combine result into a dataframe\n",
    "aa_data = {'Estimate': np.array(lmod.params[:]), 'SE': min_se, \n",
    "           'tStats': aa_tval, 'pValue': aa_pval}\n",
    "rows = lmod.params.index.values\n",
    "aa_tbl = pd.DataFrame(data=aa_data, index=rows)\n",
    "print(aa_tbl)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original 0.9992020497671739\n",
      "Minimum:  0.2695\n",
      "Mean:  1.0041\n",
      "Median:  0.9871\n",
      "Maximum:  2.3217\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0.5,1,'Distribution Plot of Estimated Sigma^2')"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# The original sigma^2\n",
    "print('Original:', sum(res**2) / (df.shape[0] - df.shape[1] - 1))\n",
    "# Sigma^2 after bootstrapping 1000 times\n",
    "print('Minimum: ', round(min(n_ss), 4))\n",
    "print('Mean: ', round(np.mean(n_ss), 4))\n",
    "print('Median: ', round(np.median(n_ss), 4))\n",
    "print('Maximum: ', round(max(n_ss), 4))\n",
    "\n",
    "# distribution plot for sigma^2\n",
    "sns.distplot(n_ss)\n",
    "plt.title('Distribution Plot of Estimated Sigma^hat^2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original 0.991400931565747\n"
     ]
    }
   ],
   "source": [
    "print('Original', cvrg_prob(sum(res**2) / (df.shape[0] - df.shape[1] - 1), alpha = 0.05))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original: 0.991400931565747\n",
      "Minimum:  0.9734862056610534\n",
      "Mean:  0.9921175206019348\n",
      "Median:  0.991400931565747\n",
      "Maximum:  0.9953421712647796\n"
     ]
    }
   ],
   "source": [
    "# Function to compute the coverage probability\n",
    "def cvrg_prob(n_ss, alpha = 0.05):\n",
    "    # input: \n",
    "    #   n_ss - sigma^2 chosen after bootstrapping\n",
    "    #   alpha - a value between [0,1] representing alpha level of confidence interval\n",
    "    # output: \n",
    "    #   the coverage probability after resampling residuals\n",
    "    \n",
    "    y = df['alcohol']\n",
    "    ci = (fitted - t.ppf(1 - alpha/2, 2783)*n_ss, \n",
    "          fitted + t.ppf(1 - alpha/2, 2783)*n_ss)\n",
    "    cvrg_prob = sum((y > ci[0]) & (y < ci[1])) / df.shape[0]\n",
    "    return(cvrg_prob)\n",
    "    \n",
    "# Test the function\n",
    "#cvrg_prob(min(n_ss), alpha = 0.05)\n",
    "\n",
    "\n",
    "# The original coverage probability\n",
    "print('Coverage Probability:')\n",
    "print('Original:', cvrg_prob(sum(res**2) / (df.shape[0] - df.shape[1] - 1), alpha = 0.05))\n",
    "\n",
    "# Coverage probability after bootstrapping 1000 times\n",
    "print('Minimum: ', cvrg_prob(min(n_ss), alpha = 0.05))\n",
    "print('Mean: ', cvrg_prob(np.mean(n_ss), alpha = 0.05))\n",
    "print('Median: ', cvrg_prob(np.median(n_ss), alpha = 0.05))\n",
    "print('Maximum: ', cvrg_prob(max(n_ss), alpha = 0.05))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Intro\n",
    "In this section, we describe a method developed by ourselves to construct new standard errors after bootstrapping residuals. We tried this method before reasearching on the common way of bootstrapping residuals and decided to put it as our additional analysis.\n",
    "\n",
    "The main differences between the core analysis and here is that we only improved the standard errors by resampling residuals. In other words, we used the coefficient estimates derived from the initial regression model instead of the new coefficient estimates updated by the resampled residuals. Compared to the core analysis, here we computed the ${\\hat{\\sigma}}^{2^*} = \\frac{\\sum{r^*}}{n-p-1}$. Then, choosing the ${\\hat{\\sigma}}^{2^*}$ from our 1000 times bootrapping. After that, we computed the standard errors, t-statistic, and p-values as what we did in the core analysis. The original thought behind this method was that we wanted to improve the standard errors without changing the point estimates."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "如何解釋選最小的SE？\n",
    "1. 選mean or median和原本的沒有差多少，使用最小的SE就是盡量不選到outlier -> 可以畫個distribution plot或是做個summary table\n",
    "2. 若從coverage probability來看，原本的覆蓋率有到99%，超過95%十分地高，若選擇最小的se，coverage probability仍然有97%，代表沒犧牲到多少，但se還變小了\n",
    "\n",
    "Why choosing the smallest SE?\n",
    "1. It appears that choosing the mean or median of bootstrapped residuals, the results weren't differ a lot compared to the original one. Hence, we chose the smallest SE to avoid residual that computed from the outlier.\n",
    "2. The coverage probability is the percentage of samples that contain the true value\n",
    "As we \n"
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
