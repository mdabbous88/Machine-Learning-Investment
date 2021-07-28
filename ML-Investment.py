#!/usr/bin/env python
# coding: utf-8

# # User input of a stock 

# In[40]:


#User input of symbol of stock. e.g:AAPL, TSLA....
symbol = input("Please enter a stock symbol:\n")


# # API request to pull data

# In[41]:


import requests
# To create an API_key, please create an account @https://rapidapi.com/apidojo/api/yahoo-finance1/
from Config import API_key
url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"

querystring = {"symbol":symbol,"region":"CANADA"}

headers = {
    'x-rapidapi-key': API_key,
    'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

response = requests.request("GET", url, headers=headers, params=querystring).json()


print(response)


# In[42]:


import sys
# This code shows how many dates are available in the json data. The length of json data changes from 1 stock 
#to another because of different events recorded other than day to day transactions (dividend, stock split ....)
#define a list that holds all the dates in a json file
year = []
# Loop through the json file, 1000 is chosen because no 1 year data is going to reach this length. Exit try-except
#once 366 rounds are finsihed
for i in range(366):
                try:
                    date = response["prices"][i]['date']
                    year.append(date)
                except:
                    sys.exit
                    
trading_days = len(year)
trading_days


# In[43]:


import numpy as np
#Year accounts for working days (365 minus weekends and holidays)
#Create lists to hold all the dates, open price, high price, low price, close price, volume of transactions, 
#and the adjusted closing price
Date = []
OpenPrice = []
HighPrice = []
LowPrice = []
ClosePrice = []
VolumeTransactions = []
adjPrice = []


#Loop throught the trading days, which is the maximum legth of a json file
for i in range(trading_days):
#try - except block is used to store null values where data is not present (dividend data for example)
                    
        try:   
            #Get the date and append the date to a list
            date = response["prices"][i]['date']
            Date.append(date)
        except:            
            #Add null when no date is found
            date = np.NAN
            Date.append(date)
    
        try:
            #Get the open price and append the price to a list
            open_price = response["prices"][i]['open']
            OpenPrice.append(open_price)
        except: 
            #Add null when no open_price is found
            open_price = np.NAN
            OpenPrice.append(open_price)

        try:  
            #Get the high price and append the price to a list
            high_price = response["prices"][i]['high']
            HighPrice.append(high_price)
        except:
            #Add null when no high_price is found
            high_price = np.NAN
            HighPrice.append(high_price)
            
        try:
            #Get the low price and append the price to a list
            low_price = response["prices"][i]['low']
            LowPrice.append(low_price)
        except:   
            #Add null when no low_price is found
            low_price = np.NAN
            LowPrice.append(low_price)
            
        try:
            #Get the close price and append the price to a list
            close_price = response["prices"][i]['close']
            ClosePrice.append(close_price)  
        except:
            #Add null when no close_price is found
            close_price = np.NAN
            ClosePrice.append(close_price)
            
        try:
            #Get the volume of transactions and append the price to a list
            volume = response["prices"][i]['volume']
            VolumeTransactions.append(volume)        
        except:
            #Add null when no volume is found
            volume = np.NAN
            VolumeTransactions.append(volume)
            
        try:
             #Get the adjPrice and append the price to a list
            adj_Price = response["prices"][i]['adjclose']
            adjPrice.append(adj_Price)
        except:
            #Add null when no adj_Price is found
            adj_Price = np.NAN
            adjPrice.append(adj_Price)  
        


# # API data Saved in a Data Frame

# In[44]:


#Join all the lists created to form a dataframe
import pandas as pd
DF= pd.DataFrame({'Market_Date': Date,'Open Price $': OpenPrice,'High Price $':HighPrice,'Low Price $': LowPrice,'Close Price $': ClosePrice,'Volume':VolumeTransactions,"Adjusted Close Price $":adjPrice})
DF


# In[45]:


import datetime
#Change all unix dates to regular dates
DF['Market_Date'] = pd.to_datetime(DF['Market_Date'], unit='s', origin='unix').dt.date
DF


# In[46]:


#Check for Null values
DF.isnull().sum()


# In[47]:


#Drop Null values if they exist and save them to a new data frame. Change index to Market_date to help join in SQL
stock_df = DF.dropna()
stock_df.set_index("Market_Date", inplace = True)
stock_df


# # Bollinger Bands Indicator

# In[48]:


#Code to create the bollinger bands
import ta
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
# Initialize Bollinger Bands Indicator
BB_df = pd.DataFrame()
indicator_bb = BollingerBands(close=stock_df["Adjusted Close Price $"], window=20, window_dev=2)
BB_df['Adjusted Close Price $_m'] = indicator_bb.bollinger_mavg()
BB_df['bb_bbh'] = indicator_bb.bollinger_hband()
BB_df['bb_bbl'] = indicator_bb.bollinger_lband()
pd.set_option("display.max_rows", None, "display.max_columns", None)
# Add Bollinger Band high indicator
BB_df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

# Add Bollinger Band low indicator
BB_df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

# Add Width Size Bollinger Bands
BB_df['bb_bbw'] = indicator_bb.bollinger_wband()

# Add Percentage Bollinger Bands
BB_df['bb_bbp'] = indicator_bb.bollinger_pband()

BB_df


# # Relative Strength Index

# In[49]:


# code to calculate RSI indicator add to the stock_df 
import warnings
warnings.filterwarnings("ignore")
RSI = ta.momentum.RSIIndicator(close=stock_df["Adjusted Close Price $"], window=20, fillna= False)
stock_df['RSI'] = RSI.rsi()


# # Moving Average

# In[50]:


#20 day moving average, add it to the stock_df 
stock_df['20d_MA'] = stock_df["Adjusted Close Price $"].rolling(20).mean()
stock_df['20d_MA']
stock_df


# # Data transition to Postgres SQL Database

# In[51]:


from os import environ
from sqlalchemy import create_engine
import getpass
Passw0rd = getpass.getpass(prompt='Password: ', stream=None) 

# Here you want to change your database, username & password according to your own values
param_dic = {
    "host"      : "localhost",
    "database"  : "Group13 Project",
    "user"      : "postgres",
    "password"  : Passw0rd
}

#establish connection with postgres sql
connect = "postgresql+psycopg2://%s:%s@%s:5432/%s" % (
    param_dic['user'],
    param_dic['password'],
    param_dic['host'],
    param_dic['database']
)
#connect to sql
engine = create_engine(connect)


# In[52]:


# Export stock_df and store it in postgres sql, each stock will have it is own stock_df
stock_df.to_sql('stock_data_'+symbol, con=engine,index=True, if_exists='replace',method='multi')


# In[53]:


# Export BB_df and store it in postgres sql, each stock will have it is own BB_df
BB_df.to_sql('bollinger_bands_'+symbol, con=engine,index=True, if_exists='replace',method='multi')


# Joining two data tables in Postgres using Pandas

# In[54]:


#result_set is a join of the stock_df and BB_df on Market_date column.
result_set = engine.execute('select * from "stock_data_'+symbol+'" inner join "bollinger_bands_'+symbol+'" using ("Market_Date")')


# In[55]:


#df_join is a dataframe established from result_set
df_join = pd.DataFrame(result_set)
df_join.columns = ['Market_date', 'Open Price $', 'High Price $','Low Price $', 'Close Price $', 'Volume','Adjusted Close Price $','RSI', '20d_MA','Adjusted Close Price $_m', 'bb_bbh','bb_bbl', 'bb_bbhi', 'bb_bbli','bb_bbw', 'bb_bbp']
df_join


# In[56]:


print(stock_df.values)


# # Machine Learning

# In[57]:


#Linear Regression Machine Learning
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[58]:


X = stock_df[['Open Price $','High Price $','Low Price $','Volume']]
X.reset_index(drop = True, inplace = True )
X.to_numpy()


# In[59]:


y = stock_df.pop('Adjusted Close Price $')
y


# In[60]:


from sklearn.model_selection import train_test_split
#train and test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[61]:


model.fit(X_train, y_train)


# In[62]:


training_score = model.score(X_train, y_train)
testing_score = model.score(X_test, y_test)
testing_score


# In[63]:


stock_df


# Close price prediction

# In[64]:


Open_price= input("Please enter an Open Price:\n")
High_price= input("Please enter an High Price:\n")
Low_price= input("Please enter an Low Price:\n")
Volume= input("Please enter an Volume:\n")

X_pred = [Open_price,High_price, Low_price, Volume]
X_pred = np.array(X_pred).reshape(1,4)
X_pred
y_pred = model.predict(X_pred)
print(f"The predicted closing price for the day is  {y_pred}")



# In[ ]:





# In[ ]:





# In[ ]:




