#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


#load the datasets
crypto_df = pd.read_csv('train.csv')
asset_df = pd.read_csv('asset_details.csv')
crypto_df = crypto_df.merge(asset_df,left_on='Asset_ID',right_on='Asset_ID')
crypto_df['date'] = pd.to_datetime(crypto_df['timestamp'],unit='s')


#subset the data by Bitcoin Cash and time intervals
bitcoin_cash_df = crypto_df.loc[crypto_df.Asset_Name == 'Bitcoin Cash',:].copy()

bitcoin_cash_df.index = bitcoin_cash_df['date']

bitcoin_cash_df_hour = bitcoin_cash_df.resample('H').mean()
bitcoin_cash_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
bitcoin_cash_df_hour = bitcoin_cash_df_hour.dropna()

bitcoin_cash_df_day = bitcoin_cash_df.resample('D').mean()
bitcoin_cash_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
bitcoin_cash_df_day = bitcoin_cash_df_day.dropna()


#subset the data by Binance Coin and time intervals
binance_coin_df = crypto_df.loc[crypto_df.Asset_Name == 'Binance Coin',:].copy()

binance_coin_df.index = binance_coin_df['date']

binance_coin_df_hour = binance_coin_df.resample('H').mean()
binance_coin_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
binance_coin_df_hour = binance_coin_df_hour.dropna()

binance_coin_df_day = binance_coin_df.resample('D').mean()
binance_coin_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
binance_coin_df_day = binance_coin_df_day.dropna()

#subset the data by Bitcoin and time intervals
bitcoin_df = crypto_df.loc[crypto_df.Asset_Name == 'Bitcoin',:].copy()

bitcoin_df.index = bitcoin_df['date']

bitcoin_df_hour = bitcoin_df.resample('H').mean()
bitcoin_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
bitcoin_df_hour = bitcoin_df_hour.dropna()

bitcoin_df_day = bitcoin_df.resample('D').mean()
bitcoin_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
bitcoin_df_day = bitcoin_df_day.dropna()

#subset the data by EOS.IO and time intervals
eos_io_df = crypto_df.loc[crypto_df.Asset_Name == 'EOS.IO',:].copy()

eos_io_df.index = eos_io_df['date']

eos_io_df_hour = eos_io_df.resample('H').mean()
eos_io_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
eos_io_df_hour = eos_io_df_hour.dropna()

eos_io_df_day = eos_io_df.resample('D').mean()
eos_io_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
eos_io_df_day = eos_io_df_day.dropna()


#subset the data by Ethereum Classic and time intervals
eth_class_df = crypto_df.loc[crypto_df.Asset_Name == 'Ethereum Classic',:].copy()

eth_class_df.index = eth_class_df['date']

eth_class_df_hour = eth_class_df.resample('H').mean()
eth_class_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
eth_class_df_hour = eth_class_df_hour.dropna()

eth_class_df_day = eth_class_df.resample('D').mean()
eth_class_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
eth_class_df_day = eth_class_df_day.dropna()


#subset the data by Ethereum and time intervals
eth_df = crypto_df.loc[crypto_df.Asset_Name == 'Ethereum',:].copy()

eth_df.index = eth_df['date']

eth_df_hour = eth_df.resample('H').mean()
eth_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
eth_df_hour = eth_df_hour.dropna()

eth_df_day = eth_df.resample('D').mean()
eth_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
eth_df_day = eth_df_day.dropna()

#subset the data by Litecoin and time intervals
litecoin_df = crypto_df.loc[crypto_df.Asset_Name == 'Litecoin',:].copy()

litecoin_df.index = litecoin_df['date']

litecoin_df_hour = litecoin_df.resample('H').mean()
litecoin_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
litecoin_df_hour = litecoin_df_hour.dropna()

litecoin_df_day = litecoin_df.resample('D').mean()
litecoin_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
litecoin_df_day = litecoin_df_day.dropna()

#subset the data by Monero and time intervals
monero_df = crypto_df.loc[crypto_df.Asset_Name == 'Monero',:].copy()

monero_df.index = monero_df['date']

monero_df_hour = monero_df.resample('H').mean()
monero_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
monero_df_hour = monero_df_hour.dropna()

monero_df_day = monero_df.resample('D').mean()
monero_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
monero_df_day = monero_df_day.dropna()

#subset the data by TRON and time intervals
TRON_df = crypto_df.loc[crypto_df.Asset_Name == 'TRON',:].copy()

TRON_df.index = TRON_df['date']

TRON_df_hour = TRON_df.resample('H').mean()
TRON_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
TRON_df_hour = TRON_df_hour.dropna()

TRON_df_day = TRON_df.resample('D').mean()
TRON_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
TRON_df_day = TRON_df_day.dropna()

#subset the data by Stellar and time intervals
stellar_df = crypto_df.loc[crypto_df.Asset_Name == 'Stellar',:].copy()

stellar_df.index = stellar_df['date']

stellar_df_hour = stellar_df.resample('H').mean()
stellar_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
stellar_df_hour = stellar_df_hour.dropna()

stellar_df_day = stellar_df.resample('D').mean()
stellar_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
stellar_df_day = stellar_df_day.dropna()

#subset the data by Cardano and time intervals
cardano_df = crypto_df.loc[crypto_df.Asset_Name == 'Cardano',:].copy()

cardano_df.index = cardano_df['date']

cardano_df_hour = cardano_df.resample('H').mean()
cardano_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
cardano_df_hour = cardano_df_hour.dropna()

cardano_df_day = cardano_df.resample('D').mean()
cardano_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
cardano_df_day = cardano_df_day.dropna()

#subset the data by IOTA and time intervals
IOTA_df = crypto_df.loc[crypto_df.Asset_Name == 'IOTA',:].copy()

IOTA_df.index = IOTA_df['date']

IOTA_df_hour = IOTA_df.resample('H').mean()
IOTA_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
IOTA_df_hour = IOTA_df_hour.dropna()

IOTA_df_day = IOTA_df.resample('D').mean()
IOTA_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
IOTA_df_day = IOTA_df_day.dropna()

#subset the data by Maker and time intervals
maker_df = crypto_df.loc[crypto_df.Asset_Name == 'Maker',:].copy()

maker_df.index = maker_df['date']

maker_df_hour = maker_df.resample('H').mean()
maker_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
maker_df_hour = maker_df_hour.dropna()

maker_df_day = maker_df.resample('D').mean()
maker_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
maker_df_day = maker_df_day.dropna()

#subset the data by Dogecoin and time intervals
dogecoin_df = crypto_df.loc[crypto_df.Asset_Name == 'Dogecoin',:].copy()

dogecoin_df.index = dogecoin_df['date']

dogecoin_df_hour = dogecoin_df.resample('H').mean()
dogecoin_df_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
dogecoin_df_hour = dogecoin_df_hour.dropna()

dogecoin_df_day = dogecoin_df.resample('D').mean()
dogecoin_df_day.replace([np.inf, -np.inf], np.nan, inplace=True)
dogecoin_df_day = dogecoin_df_day.dropna()


print('Your data is ready')


# In[59]:


#create the Bitcoin Cash models
import statsmodels
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#create our models
#define a function to create a dataframe which returns statistics from the ols model


def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })


#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
bitcoin_cash_hour1 = smf.ols(f1,data=bitcoin_cash_df_hour).fit()
bitcoin_cash_hour2 = smf.ols(f2,data=bitcoin_cash_df_hour).fit()
bitcoin_cash_hour3 = smf.ols(f3,data=bitcoin_cash_df_hour).fit()
bitcoin_cash_hour4 = smf.ols(f4,data=bitcoin_cash_df_hour).fit()
bitcoin_cash_hour5 = smf.ols(f5,data=bitcoin_cash_df_hour).fit()
bitcoin_cash_day1 = smf.ols(f1,data=bitcoin_cash_df_day).fit()
bitcoin_cash_day2 = smf.ols(f2,data=bitcoin_cash_df_day).fit()
bitcoin_cash_day3 = smf.ols(f3,data=bitcoin_cash_df_day).fit()
bitcoin_cash_day4 = smf.ols(f4,data=bitcoin_cash_df_day).fit()
bitcoin_cash_day5 = smf.ols(f5,data=bitcoin_cash_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(bitcoin_cash_hour1,bitcoin_cash_hour2,bitcoin_cash_hour3,bitcoin_cash_hour4,bitcoin_cash_hour5,                       bitcoin_cash_day1,bitcoin_cash_day2,bitcoin_cash_day3,bitcoin_cash_day4,bitcoin_cash_day5)
bitcoin_cash_idx = ['bitcoin_cash_hour1','bitcoin_cash_hour2','bitcoin_cash_hour3','bitcoin_cash_hour4','bitcoin_cash_hour5',                   'bitcoin_cash_day1','bitcoin_cash_day2','bitcoin_cash_day3','bitcoin_cash_day4','bitcoin_cash_day5']
ols_tables.index = bitcoin_cash_idx

#return the results
ols_tables


# In[60]:


#chart our bitcoin cash model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

bitcoin_cash_results = pd.concat([bitcoin_cash_hour1.params,bitcoin_cash_hour2.params,bitcoin_cash_hour3.params,bitcoin_cash_hour4.params,
                                  bitcoin_cash_hour5.params],axis=1).rename(columns=lambda x: 'bitcoin_cash_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=bitcoin_cash_results,
                   dodge=True, join=False)

plt.tight_layout()

bitcoin_cash_results = pd.concat([bitcoin_cash_day1.params,bitcoin_cash_day2.params,bitcoin_cash_day3.params,
                                  bitcoin_cash_day4.params,bitcoin_cash_day5.params ],axis=1).rename(columns=lambda x: 'bitcoin_cash_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=bitcoin_cash_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[61]:


#create our anova tables for bitcoin cash model
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
bitcoin_cash_anova = sm.stats.anova_lm(bitcoin_cash_hour1,bitcoin_cash_hour2,bitcoin_cash_hour3,bitcoin_cash_hour4,bitcoin_cash_hour5,                       bitcoin_cash_day1,bitcoin_cash_day2,bitcoin_cash_day3,bitcoin_cash_day4,bitcoin_cash_day5)
bitcoin_cash_anova.index = bitcoin_cash_idx
bitcoin_cash_anova


# In[82]:


#create the Binance Coin models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
binance_coin_hour1 = smf.ols(f1,data=binance_coin_df_hour).fit()
binance_coin_hour2 = smf.ols(f2,data=binance_coin_df_hour).fit()
binance_coin_hour3 = smf.ols(f3,data=binance_coin_df_hour).fit()
binance_coin_hour4 = smf.ols(f4,data=binance_coin_df_hour).fit()
binance_coin_hour5 = smf.ols(f5,data=binance_coin_df_hour).fit()
binance_coin_day1 = smf.ols(f1,data=binance_coin_df_day).fit()
binance_coin_day2 = smf.ols(f2,data=binance_coin_df_day).fit()
binance_coin_day3 = smf.ols(f3,data=binance_coin_df_day).fit()
binance_coin_day4 = smf.ols(f4,data=binance_coin_df_day).fit()
binance_coin_day5 = smf.ols(f5,data=binance_coin_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(binance_coin_hour1,binance_coin_hour2,binance_coin_hour3,binance_coin_hour4,binance_coin_hour5,                      binance_coin_day1,binance_coin_day2,binance_coin_day3,binance_coin_day4,binance_coin_day5)
binance_coin_idx = ['binance_coin_hour1','binance_coin_hour2','binance_coin_hour3','binance_coin_hour4','binance_coin_hour5',                   'binance_coin_day1','binance_coin_day2','binance_coin_day3','binance_coin_day4','binance_coin_day5']
ols_tables.index = binance_coin_idx

#return the results
ols_tables


# In[83]:


#chart our binance coin model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

binance_coin_results = pd.concat([binance_coin_hour1.params,binance_coin_hour2.params,binance_coin_hour3.params,binance_coin_hour4.params,                                  binance_coin_hour5.params],axis=1).rename(columns=lambda x: 'binance_coin_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=binance_coin_results,
                   dodge=True, join=False)

plt.tight_layout()

binance_coin_results = pd.concat([binance_coin_day1.params,binance_coin_day2.params,binance_coin_day3.params,binance_coin_day4.params,binance_coin_day5.params],                                 axis=1).rename(columns=lambda x: 'binance_coin_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=binance_coin_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[84]:


#create our anova tables for binance models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
binance_coin_anova = sm.stats.anova_lm(binance_coin_hour1,binance_coin_hour2,binance_coin_hour3,binance_coin_hour4,                                       binance_coin_hour5,binance_coin_day1,binance_coin_day2,binance_coin_day3,                                       binance_coin_day4,binance_coin_day5)
binance_coin_anova.index = binance_coin_idx
binance_coin_anova


# In[85]:


#create the Bitcoin models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
bitcoin_hour1 = smf.ols(f1,data=bitcoin_df_hour).fit()
bitcoin_hour2 = smf.ols(f2,data=bitcoin_df_hour).fit()
bitcoin_hour3 = smf.ols(f3,data=bitcoin_df_hour).fit()
bitcoin_hour4 = smf.ols(f4,data=bitcoin_df_hour).fit()
bitcoin_hour5 = smf.ols(f5,data=bitcoin_df_hour).fit()
bitcoin_day1 = smf.ols(f1,data=bitcoin_df_day).fit()
bitcoin_day2 = smf.ols(f2,data=bitcoin_df_day).fit()
bitcoin_day3 = smf.ols(f3,data=bitcoin_df_day).fit()
bitcoin_day4 = smf.ols(f4,data=bitcoin_df_day).fit()
bitcoin_day5 = smf.ols(f5,data=bitcoin_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(bitcoin_hour1,bitcoin_hour2,bitcoin_hour3,bitcoin_hour4,bitcoin_hour5,                       bitcoin_day1,bitcoin_day2,bitcoin_day3,bitcoin_day4,bitcoin_day5)
bitcoin_idx = ['bitcoin_hour1','bitcoin_hour2','bitcoin_hour3','bitcoin_hour4','bitcoin_hour5',              'bitcoin_day1','bitcoin_day2','bitcoin_day3','bitcoin_day4','bitcoin_day5']
ols_tables.index = bitcoin_idx

#return the results
ols_tables


# In[86]:


#chart our bitcoin model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

bitcoin_results = pd.concat([bitcoin_hour1.params,bitcoin_hour2.params,bitcoin_hour3.params,bitcoin_hour4.params,bitcoin_hour5.params],                            axis=1).rename(columns=lambda x: 'bitcoin_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=bitcoin_results,
                   dodge=True, join=False)

plt.tight_layout()

bitcoin_results = pd.concat([bitcoin_day1.params,bitcoin_day2.params,bitcoin_day3.params,bitcoin_day4.params,bitcoin_day5.params],                            axis=1).rename(columns=lambda x: 'bitcoin_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=bitcoin_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[87]:


#create our anova tables for bitcoin models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
bitcoin_anova = sm.stats.anova_lm(bitcoin_hour1,bitcoin_hour2,bitcoin_hour3,bitcoin_hour4,bitcoin_hour5,                       bitcoin_day1,bitcoin_day2,bitcoin_day3,bitcoin_day4,bitcoin_day5)
bitcoin_anova.index = bitcoin_idx
bitcoin_anova


# In[89]:


#create the EOS.IO models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
eos_io_hour1 = smf.ols(f1,data=eos_io_df_hour).fit()
eos_io_hour2 = smf.ols(f2,data=eos_io_df_hour).fit()
eos_io_hour3 = smf.ols(f3,data=eos_io_df_hour).fit()
eos_io_hour4 = smf.ols(f4,data=eos_io_df_hour).fit()
eos_io_hour5 = smf.ols(f5,data=eos_io_df_hour).fit()
eos_io_day1 = smf.ols(f1,data=eos_io_df_day).fit()
eos_io_day2 = smf.ols(f2,data=eos_io_df_day).fit()
eos_io_day3 = smf.ols(f3,data=eos_io_df_day).fit()
eos_io_day4 = smf.ols(f4,data=eos_io_df_day).fit()
eos_io_day5 = smf.ols(f5,data=eos_io_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(eos_io_hour1,eos_io_hour2,eos_io_hour3,eos_io_hour4,eos_io_hour5,                      eos_io_day1,eos_io_day2,eos_io_day3,eos_io_day4,eos_io_day5)
eos_io_idx = ['eos_io_hour1','eos_io_hour2','eos_io_hour3','eos_io_hour4','eos_io_hour5',             'eos_io_day1','eos_io_day2','eos_io_day3','eos_io_day4','eos_io_day5']
ols_tables.index = eos_io_idx

#return the results
ols_tables


# In[90]:


#chart our EOS.IO model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

eos_io_results = pd.concat([eos_io_hour1.params,eos_io_hour2.params,eos_io_hour3.params,eos_io_hour4.params,eos_io_hour5.params],                           axis=1).rename(columns=lambda x: 'eos_io_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=eos_io_results,
                   dodge=True, join=False)

plt.tight_layout()

eos_io_results = pd.concat([eos_io_day1.params,eos_io_day2.params,eos_io_day3.params,eos_io_day4.params,eos_io_day5.params],                              axis=1).rename(columns=lambda x: 'eos_io_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=eos_io_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[91]:


#create our anova tables for EOS.IO models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
eos_io_anova = sm.stats.anova_lm(eos_io_hour1,eos_io_hour2,eos_io_hour3,eos_io_hour4,eos_io_hour5,                      eos_io_day1,eos_io_day2,eos_io_day3,eos_io_day4,eos_io_day5)
eos_io_anova.index = eos_io_idx
eos_io_anova


# In[95]:


#create the Ethereum Classic models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
eth_class_hour1 = smf.ols(f1,data=eth_class_df_hour).fit()
eth_class_hour2 = smf.ols(f2,data=eth_class_df_hour).fit()
eth_class_hour3 = smf.ols(f3,data=eth_class_df_hour).fit()
eth_class_hour4 = smf.ols(f4,data=eth_class_df_hour).fit()
eth_class_hour5 = smf.ols(f5,data=eth_class_df_hour).fit()
eth_class_day1 = smf.ols(f1,data=eth_class_df_day).fit()
eth_class_day2 = smf.ols(f2,data=eth_class_df_day).fit()
eth_class_day3 = smf.ols(f3,data=eth_class_df_day).fit()
eth_class_day4 = smf.ols(f4,data=eth_class_df_day).fit()
eth_class_day5 = smf.ols(f5,data=eth_class_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(eth_class_hour1,eth_class_hour2,eth_class_hour3,eth_class_hour4,eth_class_hour5,                      eth_class_day1,eth_class_day2,eth_class_day3,eth_class_day4,eth_class_day5)
eth_class_idx = ['eth_class_hour1','eth_class_hour2','eth_class_hour3','eth_class_hour4','eth_class_hour5',                'eth_class_day1','eth_class_day2','eth_class_day3','eth_class_day4','eth_class_day5']
ols_tables.index = eth_class_idx


#return the results
ols_tables


# In[96]:


#chart our Ethereum Classic model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

eth_class_results = pd.concat([eth_class_hour1.params,eth_class_hour2.params,eth_class_hour3.params,eth_class_hour4.params,eth_class_hour5.params],                              axis=1).rename(columns=lambda x: 'eth_class_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=eth_class_results,
                   dodge=True, join=False)

plt.tight_layout()

eth_class_results = pd.concat([eth_class_day1.params,eth_class_day2.params,eth_class_day3.params,eth_class_day4.params,eth_class_day5.params],                              axis=1).rename(columns=lambda x: 'eth_class_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=eth_class_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[97]:


#create our anova tables for Ethereum Classic models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
eth_class_anova = sm.stats.anova_lm(eth_class_hour1,eth_class_hour2,eth_class_hour3,eth_class_hour4,eth_class_hour5,                      eth_class_day1,eth_class_day2,eth_class_day3,eth_class_day4,eth_class_day5)
eth_class_anova.index = eth_class_idx
eth_class_anova


# In[98]:


#create the Ethereum models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
eth_hour1 = smf.ols(f1,data=eth_df_hour).fit()
eth_hour2 = smf.ols(f2,data=eth_df_hour).fit()
eth_hour3 = smf.ols(f3,data=eth_df_hour).fit()
eth_hour4 = smf.ols(f4,data=eth_df_hour).fit()
eth_hour5 = smf.ols(f5,data=eth_df_hour).fit()
eth_day1 = smf.ols(f1,data=eth_df_day).fit()
eth_day2 = smf.ols(f2,data=eth_df_day).fit()
eth_day3 = smf.ols(f3,data=eth_df_day).fit()
eth_day4 = smf.ols(f4,data=eth_df_day).fit()
eth_day5 = smf.ols(f5,data=eth_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(eth_hour1,eth_hour2,eth_hour3,eth_hour4,eth_hour5,                       eth_day1,eth_day2,eth_day3,eth_day4,eth_day5)
eth_idx = ['eth_hour1','eth_hour2','eth_hour3','eth_hour4','eth_hour5',           'eth_day1','eth_day2','eth_day3','eth_day4','eth_day5']
ols_tables.index = eth_idx

#return the results
ols_tables


# In[125]:


#chart our Ethereum model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

eth_results = pd.concat([eth_hour1.params,eth_hour2.params,eth_hour3.params,eth_hour4.params,eth_hour5.params],                        axis=1).rename(columns=lambda x: 'eth_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=eth_results,
                   dodge=True, join=False)

plt.tight_layout()

eth_results = pd.concat([eth_day1.params,eth_day2.params,eth_day3.params,eth_day4.params,eth_day5.params],                        axis=1).rename(columns=lambda x: 'eth_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=eth_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[100]:


#create our anova tables for Ethereum models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
eth_anova = sm.stats.anova_lm(eth_hour1,eth_hour2,eth_hour3,eth_hour4,eth_hour5,                       eth_day1,eth_day2,eth_day3,eth_day4,eth_day5)
eth_anova.index = eth_idx
eth_anova


# In[101]:


#create the Litecoin models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models],
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
litecoin_hour1 = smf.ols(f1,data=litecoin_df_hour).fit()
litecoin_hour2 = smf.ols(f2,data=litecoin_df_hour).fit()
litecoin_hour3 = smf.ols(f3,data=litecoin_df_hour).fit()
litecoin_hour4 = smf.ols(f4,data=litecoin_df_hour).fit()
litecoin_hour5 = smf.ols(f5,data=litecoin_df_hour).fit()
litecoin_day1 = smf.ols(f1,data=litecoin_df_day).fit()
litecoin_day2 = smf.ols(f2,data=litecoin_df_day).fit()
litecoin_day3 = smf.ols(f3,data=litecoin_df_day).fit()
litecoin_day4 = smf.ols(f4,data=litecoin_df_day).fit()
litecoin_day5 = smf.ols(f5,data=litecoin_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(litecoin_hour1,litecoin_hour2,litecoin_hour3,litecoin_hour4,litecoin_hour5,                      litecoin_day1,litecoin_day2,litecoin_day3,litecoin_day4,litecoin_day5)
litecoin_idx = ['litecoin_hour1','litecoin_hour2','litecoin_hour3','litecoin_hour4','litecoin_hour5',               'litecoin_day1','litecoin_day2','litecoin_day3','litecoin_day4','litecoin_day5']
ols_tables.index = litecoin_idx

#return the results
ols_tables


# In[102]:


#chart our Litecoin model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

litecoin_results = pd.concat([litecoin_hour1.params,litecoin_hour2.params,litecoin_hour3.params,litecoin_hour4.params,litecoin_hour5.params],                              axis=1).rename(columns=lambda x: 'litecoin_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=litecoin_results,
                   dodge=True, join=False)

plt.tight_layout()

litecoin_results = pd.concat([litecoin_day1.params,litecoin_day2.params,litecoin_day3.params,litecoin_day4.params,litecoin_day5.params],                              axis=1).rename(columns=lambda x: 'litecoin_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=litecoin_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[103]:


#create our anova tables for Litecoin models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
litecoin_anova = sm.stats.anova_lm(litecoin_hour1,litecoin_hour2,litecoin_hour3,litecoin_hour4,litecoin_hour5,                      litecoin_day1,litecoin_day2,litecoin_day3,litecoin_day4,litecoin_day5)
litecoin_anova.index = litecoin_idx
litecoin_anova


# In[104]:


#create the Monero models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
monero_hour1 = smf.ols(f1,data=monero_df_hour).fit()
monero_hour2 = smf.ols(f2,data=monero_df_hour).fit()
monero_hour3 = smf.ols(f3,data=monero_df_hour).fit()
monero_hour4 = smf.ols(f4,data=monero_df_hour).fit()
monero_hour5 = smf.ols(f5,data=monero_df_hour).fit()
monero_day1 = smf.ols(f1,data=monero_df_day).fit()
monero_day2 = smf.ols(f2,data=monero_df_day).fit()
monero_day3 = smf.ols(f3,data=monero_df_day).fit()
monero_day4 = smf.ols(f4,data=monero_df_day).fit()
monero_day5 = smf.ols(f5,data=monero_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(monero_hour1,monero_hour2,monero_hour3,monero_hour4,monero_hour5,                      monero_day1,monero_day2,monero_day3,monero_day4,monero_day5)
monero_idx = ['monero_hour1','monero_hour2','monero_hour3','monero_hour4','monero_hour5',             'monero_day1','monero_day2','monero_day3','monero_day4','monero_day5']
ols_tables.index = monero_idx

#return the results
ols_tables


# In[105]:


#chart our Monero model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

monero_results = pd.concat([monero_hour1.params,monero_hour2.params,monero_hour3.params,monero_hour4.params,monero_hour5.params],                           axis=1).rename(columns=lambda x: 'monero_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=monero_results,
                   dodge=True, join=False)

plt.tight_layout()

monero_results = pd.concat([monero_day1.params,monero_day2.params,monero_day3.params,monero_day4.params,monero_day5.params],                              axis=1).rename(columns=lambda x: 'monero_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=monero_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[106]:


#create our anova tables for Monero models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
monero_anova = sm.stats.anova_lm(monero_hour1,monero_hour2,monero_hour3,monero_hour4,monero_hour5,                      monero_day1,monero_day2,monero_day3,monero_day4,monero_day5)
monero_anova.index = monero_idx
monero_anova


# In[107]:


#create the TRON models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
TRON_hour1 = smf.ols(f1,data=TRON_df_hour).fit()
TRON_hour2 = smf.ols(f2,data=TRON_df_hour).fit()
TRON_hour3 = smf.ols(f3,data=TRON_df_hour).fit()
TRON_hour4 = smf.ols(f4,data=TRON_df_hour).fit()
TRON_hour5 = smf.ols(f5,data=TRON_df_hour).fit()
TRON_day1 = smf.ols(f1,data=TRON_df_day).fit()
TRON_day2 = smf.ols(f2,data=TRON_df_day).fit()
TRON_day3 = smf.ols(f3,data=TRON_df_day).fit()
TRON_day4 = smf.ols(f4,data=TRON_df_day).fit()
TRON_day5 = smf.ols(f5,data=TRON_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(TRON_hour1,TRON_hour2,TRON_hour3,TRON_hour4,TRON_hour5,                      TRON_day1,TRON_day2,TRON_day3,TRON_day4,TRON_day5)
TRON_idx = ['TRON_hour1','TRON_hour2','TRON_hour3','TRON_hour4','TRON_hour5',           'TRON_day1','TRON_day2','TRON_day3','TRON_day4','TRON_day5']
ols_tables.index = TRON_idx

#return the results
ols_tables


# In[108]:


#chart our TRON model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

TRON_results = pd.concat([TRON_hour1.params,TRON_hour2.params,TRON_hour3.params,TRON_hour4.params,TRON_hour5.params],                         axis=1).rename(columns=lambda x: 'TRON_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=TRON_results,
                   dodge=True, join=False)

plt.tight_layout()

TRON_results = pd.concat([TRON_day1.params,TRON_day2.params,TRON_day3.params,TRON_day4.params,TRON_day5.params],                         axis=1).rename(columns=lambda x: 'TRON_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=TRON_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[109]:


#create our anova tables for TRON models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
TRON_anova = sm.stats.anova_lm(TRON_hour1,TRON_hour2,TRON_hour3,TRON_hour4,TRON_hour5,                      TRON_day1,TRON_day2,TRON_day3,TRON_day4,TRON_day5)
TRON_anova.index = TRON_idx
TRON_anova


# In[110]:


#create the Stellar models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'


#create our ols models
stellar_hour1 = smf.ols(f1,data=stellar_df_hour).fit()
stellar_hour2 = smf.ols(f2,data=stellar_df_hour).fit()
stellar_hour3 = smf.ols(f3,data=stellar_df_hour).fit()
stellar_hour4 = smf.ols(f4,data=stellar_df_hour).fit()
stellar_hour5 = smf.ols(f5,data=stellar_df_hour).fit()
stellar_day1 = smf.ols(f1,data=stellar_df_day).fit()
stellar_day2 = smf.ols(f2,data=stellar_df_day).fit()
stellar_day3 = smf.ols(f3,data=stellar_df_day).fit()
stellar_day4 = smf.ols(f4,data=stellar_df_day).fit()
stellar_day5 = smf.ols(f5,data=stellar_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(stellar_hour1,stellar_hour2,stellar_hour3,stellar_hour4,stellar_hour5,                      stellar_day1,stellar_day2,stellar_day3,stellar_day4,stellar_day5)
stellar_idx = ['stellar_hour1','stellar_hour2','stellar_hour3','stellar_hour4','stellar_hour5',              'stellar_day1','stellar_day2','stellar_day3','stellar_day4','stellar_day5']
ols_tables.index = stellar_idx

#return the results
ols_tables


# In[111]:


#chart our Stellar model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

stellar_results = pd.concat([stellar_hour1.params,stellar_hour2.params,stellar_hour3.params,stellar_hour4.params,stellar_hour5.params],                            axis=1).rename(columns=lambda x: 'stellar_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=stellar_results,
                   dodge=True, join=False)

plt.tight_layout()


stellar_results = pd.concat([stellar_day1.params,stellar_day2.params,stellar_day3.params,stellar_day4.params,stellar_day5.params],                            axis=1).rename(columns=lambda x: 'stellar_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=stellar_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[112]:


#create our anova tables for Stellar models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
stellar_anova = sm.stats.anova_lm(stellar_hour1,stellar_hour2,stellar_hour3,stellar_hour4,stellar_hour5,                      stellar_day1,stellar_day2,stellar_day3,stellar_day4,stellar_day5)
stellar_anova.index = stellar_idx
stellar_anova


# In[113]:


#create the Cardano models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
cardano_hour1 = smf.ols(f1,data=cardano_df_hour).fit()
cardano_hour2 = smf.ols(f2,data=cardano_df_hour).fit()
cardano_hour3 = smf.ols(f3,data=cardano_df_hour).fit()
cardano_hour4 = smf.ols(f4,data=cardano_df_hour).fit()
cardano_hour5 = smf.ols(f5,data=cardano_df_hour).fit()
cardano_day1 = smf.ols(f1,data=cardano_df_day).fit()
cardano_day2 = smf.ols(f2,data=cardano_df_day).fit()
cardano_day3 = smf.ols(f3,data=cardano_df_day).fit()
cardano_day4 = smf.ols(f4,data=cardano_df_day).fit()
cardano_day5 = smf.ols(f5,data=cardano_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(cardano_hour1,cardano_hour2,cardano_hour3,cardano_hour4,cardano_hour5,                      cardano_day1,cardano_day2,cardano_day3,cardano_day4,cardano_day5)
cardano_idx = ['cardano_hour1','cardano_hour2','cardano_hour3','cardano_hour4','cardano_hour5',              'cardano_day1','cardano_day2','cardano_day3','cardano_day4','cardano_day5']
ols_tables.index = cardano_idx

#return the results
ols_tables


# In[114]:


#chart our Cardano model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

cardano_results = pd.concat([cardano_hour1.params,cardano_hour2.params,cardano_hour3.params,cardano_hour4.params,cardano_hour5.params],                            axis=1).rename(columns=lambda x: 'cardano_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=cardano_results,
                   dodge=True, join=False)

plt.tight_layout()

cardano_results = pd.concat([cardano_day1.params,cardano_day2.params,cardano_day3.params,cardano_day4.params,cardano_day5.params],                              axis=1).rename(columns=lambda x: 'cardano_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=cardano_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[115]:


#create our anova tables for Cardano models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
cardano_anova = sm.stats.anova_lm(cardano_hour1,cardano_hour2,cardano_hour3,cardano_hour4,cardano_hour5,                      cardano_day1,cardano_day2,cardano_day3,cardano_day4,cardano_day5)
cardano_anova.index = cardano_idx
cardano_anova


# In[116]:


#create the IOTA models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
IOTA_hour1 = smf.ols(f1,data=IOTA_df_hour).fit()
IOTA_hour2 = smf.ols(f2,data=IOTA_df_hour).fit()
IOTA_hour3 = smf.ols(f3,data=IOTA_df_hour).fit()
IOTA_hour4 = smf.ols(f4,data=IOTA_df_hour).fit()
IOTA_hour5 = smf.ols(f5,data=IOTA_df_hour).fit()
IOTA_day1 = smf.ols(f1,data=IOTA_df_day).fit()
IOTA_day2 = smf.ols(f2,data=IOTA_df_day).fit()
IOTA_day3 = smf.ols(f3,data=IOTA_df_day).fit()
IOTA_day4 = smf.ols(f4,data=IOTA_df_day).fit()
IOTA_day5 = smf.ols(f5,data=IOTA_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(IOTA_hour1,IOTA_hour2,IOTA_hour3,IOTA_hour4,IOTA_hour5,                      IOTA_day1,IOTA_day2,IOTA_day3,IOTA_day4,IOTA_day5)
IOTA_idx = ['IOTA_hour1','IOTA_hour2','IOTA_hour3','IOTA_hour4','IOTA_hour5',           'IOTA_day1','IOTA_day2','IOTA_day3','IOTA_day4','IOTA_day5']
ols_tables.index = IOTA_idx

#return the results
ols_tables


# In[117]:


#chart our IOTA model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

IOTA_results = pd.concat([IOTA_hour1.params,IOTA_hour2.params,IOTA_hour3.params,IOTA_hour4.params,IOTA_hour5.params],                         axis=1).rename(columns=lambda x: 'IOTA_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=IOTA_results,
                   dodge=True, join=False)

plt.tight_layout()

IOTA_results = pd.concat([IOTA_day1.params,IOTA_day2.params,IOTA_day3.params,IOTA_day4.params,IOTA_day5.params],                         axis=1).rename(columns=lambda x: 'IOTA_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=IOTA_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[118]:


#create our anova tables for IOTA models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
IOTA_anova = sm.stats.anova_lm(IOTA_hour1,IOTA_hour2,IOTA_hour3,IOTA_hour4,IOTA_hour5,                      IOTA_day1,IOTA_day2,IOTA_day3,IOTA_day4,IOTA_day5)
IOTA_anova.index = IOTA_idx
IOTA_anova


# In[119]:


#create the Maker models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
maker_hour1 = smf.ols(f1,data=maker_df_hour).fit()
maker_hour2 = smf.ols(f2,data=maker_df_hour).fit()
maker_hour3 = smf.ols(f3,data=maker_df_hour).fit()
maker_hour4 = smf.ols(f4,data=maker_df_hour).fit()
maker_hour5 = smf.ols(f5,data=maker_df_hour).fit()
maker_day1 = smf.ols(f1,data=maker_df_day).fit()
maker_day2 = smf.ols(f2,data=maker_df_day).fit()
maker_day3 = smf.ols(f3,data=maker_df_day).fit()
maker_day4 = smf.ols(f4,data=maker_df_day).fit()
maker_day5 = smf.ols(f5,data=maker_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(maker_hour1,maker_hour2,maker_hour3,maker_hour4,maker_hour5,                      maker_day1,maker_day2,maker_day3,maker_day4,maker_day5)
maker_idx = ['maker_hour1','maker_hour2','maker_hour3','maker_hour4','maker_hour5',            'maker_day1','maker_day2','maker_day3','maker_day4','maker_day5']
ols_tables.index = maker_idx

#return the results
ols_tables


# In[120]:


#chart our Maker model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

maker_results = pd.concat([maker_hour1.params,maker_hour2.params,maker_hour3.params,maker_hour4.params,maker_hour5.params],                          axis=1).rename(columns=lambda x: 'maker_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=maker_results,
                   dodge=True, join=False)

plt.tight_layout()

maker_results = pd.concat([maker_day1.params,maker_day2.params,maker_day3.params,maker_day4.params,maker_day5.params],                          axis=1).rename(columns=lambda x: 'maker_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=maker_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[121]:


#create our anova tables for Maker models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
maker_anova = sm.stats.anova_lm(maker_hour1,maker_hour2,maker_hour3,maker_hour4,maker_hour5,                      maker_day1,maker_day2,maker_day3,maker_day4,maker_day5)
maker_anova.index = maker_idx
maker_anova


# In[122]:


#create the Dogecoin models
import statsmodels
import statsmodels.formula.api as smf
import numpy as np

#create our models
#define a function to create a dataframe which returns statistics from the ols model

def ols_table(*models):
    return pd.DataFrame({
        'F-pvalues':[i.f_pvalue for i in models],
        'AIC':[i.aic for i in models],
        'BIC':[i.bic for i in models],
        'Residuals':[i.df_resid for i in models],
        'R-Squared':[i.rsquared for i in models],
        'R-Squared Adj':[i.rsquared_adj for i in models]
    })

#create covariates for the model
f1 = 'Target ~ Open*Volume + Close*Volume'
f2 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume'
f3 = 'Target ~ Open*Volume + High*Volume + Low*Volume + Close*Volume + VWAP'
f4 = 'Target ~ Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP'
f5 = 'Target ~ Count + Open*VWAP + High*VWAP + Low*VWAP + Close*VWAP + Volume'

#create our ols models
dogecoin_hour1 = smf.ols(f1,data=dogecoin_df_hour).fit()
dogecoin_hour2 = smf.ols(f2,data=dogecoin_df_hour).fit()
dogecoin_hour3 = smf.ols(f3,data=dogecoin_df_hour).fit()
dogecoin_hour4 = smf.ols(f4,data=dogecoin_df_hour).fit()
dogecoin_hour5 = smf.ols(f5,data=dogecoin_df_hour).fit()
dogecoin_day1 = smf.ols(f1,data=dogecoin_df_day).fit()
dogecoin_day2 = smf.ols(f2,data=dogecoin_df_day).fit()
dogecoin_day3 = smf.ols(f3,data=dogecoin_df_day).fit()
dogecoin_day4 = smf.ols(f4,data=dogecoin_df_day).fit()
dogecoin_day5 = smf.ols(f5,data=dogecoin_df_day).fit()


#create a variable to house the dataframe and run the ols_table function with our 1 min models
#create a list with the model names
#assign the index with the model names
ols_tables = ols_table(dogecoin_hour1,dogecoin_hour2,dogecoin_hour3,dogecoin_hour4,dogecoin_hour5,                      dogecoin_day1,dogecoin_day2,dogecoin_day3,dogecoin_day4,dogecoin_day5)
dogecoin_idx = ['dogecoin_hour1','dogecoin_hour2','dogecoin_hour3','dogecoin_hour4','dogecoin_hour5',               'dogecoin_day1','dogecoin_day2','dogecoin_day3','dogecoin_day4','dogecoin_day5']
ols_tables.index = dogecoin_idx


#return the results
ols_tables


# In[123]:


#chart our Dogecoin model results
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

dogecoin_results = pd.concat([dogecoin_hour1.params,dogecoin_hour2.params,dogecoin_hour3.params,dogecoin_hour4.params,dogecoin_hour5.params],                             axis=1).rename(columns=lambda x: 'dogecoin_hour' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=dogecoin_results,
                   dodge=True, join=False)

plt.tight_layout()


dogecoin_results = pd.concat([dogecoin_day1.params,dogecoin_day2.params,dogecoin_day3.params,dogecoin_day4.params,dogecoin_day5.params],                              axis=1).rename(columns=lambda x: 'dogecoin_day' + str(x+1)).reset_index().rename(columns={'index':'param'}).melt(id_vars='param',var_name='model',value_name='estimate')

fig,ax = plt.subplots()
ax = sns.pointplot(x='estimate',y='param',hue='model', data=dogecoin_results,
                   dodge=True, join=False)

plt.tight_layout()


# In[124]:


#create our anova tables for Dogecoin models
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
dogecoin_anova = sm.stats.anova_lm(dogecoin_hour1,dogecoin_hour2,dogecoin_hour3,dogecoin_hour4,dogecoin_hour5,                      dogecoin_day1,dogecoin_day2,dogecoin_day3,dogecoin_day4,dogecoin_day5)
dogecoin_anova.index = dogecoin_idx
dogecoin_anova


# In[ ]:




