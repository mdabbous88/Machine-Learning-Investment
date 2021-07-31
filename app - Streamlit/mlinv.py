#!/usr/bin/env python
# coding: utf-8

import requests
import sys
from config import API_key
from config import db_pass
import numpy as np
import pandas as pd
import datetime
import ta
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
from os import environ
from sqlalchemy import create_engine
import getpass
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

param_dic = {
    "host"      : "localhost",
    "database"  : "Group13 Project",
    "user"      : "postgres",
    "password"  : db_pass
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

def runAll(symbol):
    response = getStockApi(symbol)
    stock_df = getStockData(response)
    BB_df = createIndicators(stock_df)
    save2DB(symbol, stock_df, BB_df)
    return getDataWithIndicator(symbol)

def getStockApi(symbol):
    # API request to pull data
    # To create an API_key, please create an account @https://rapidapi.com/apidojo/api/yahoo-finance1/

    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"

    querystring = {"symbol":symbol,"region":"US"}

    headers = {
        'x-rapidapi-key': API_key,
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
        }

    response = requests.request("GET", url, headers=headers, params=querystring).json()
    return response

def getStockData(response):
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
    #Join all the lists created to form a dataframe    
    DF= pd.DataFrame({'Market_Date': Date,'Open Price $': OpenPrice,'High Price $':HighPrice,'Low Price $': LowPrice,'Close Price $': ClosePrice,'Volume':VolumeTransactions,"Adjusted Close Price $":adjPrice})

    #Change all unix dates to regular dates
    DF['Market_Date'] = pd.to_datetime(DF['Market_Date'], unit='s', origin='unix').dt.date

    # #Check for Null values
    # DF.isnull().sum()

    #Drop Null values if they exist and save them to a new data frame. Change index to Market_date to help join in SQL
    stock_df = DF.dropna()
    stock_df.set_index("Market_Date", inplace = True)

    return stock_df

def createIndicators(stock_df):
    # Bollinger Bands Indicator
    #Code to create the bollinger bands
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

    # # Relative Strength Index

    # code to calculate RSI indicator add to the stock_df 
    
    warnings.filterwarnings("ignore")
    RSI = ta.momentum.RSIIndicator(close=stock_df["Adjusted Close Price $"], window=20, fillna= False)
    stock_df['RSI'] = RSI.rsi()

    # # Moving Average

    #20 day moving average, add it to the stock_df 
    stock_df['20d_MA'] = stock_df["Adjusted Close Price $"].rolling(20).mean()
    stock_df['20d_MA']

    # # Data transition to Postgres SQL Database
    # Passw0rd = getpass.getpass(prompt='Password: ', stream=None) 
    # Here you want to change your database, username & password according to your own values  
    return stock_df

def save2DB(symbol, stock_df, BB_df):
    # Export stock_df and store it in postgres sql, each stock will have it is own stock_df
    stock_df.to_sql('stock_data_'+symbol, con=engine,index=True, if_exists='replace',method='multi')
    # Export BB_df and store it in postgres sql, each stock will have it is own BB_df
    BB_df.to_sql('bollinger_bands_'+symbol, con=engine,index=True, if_exists='replace',method='multi')
    # Joining two data tables in Postgres using Pandas
    print('saved')

def getDataFromDB(symbol):
    #result_set is a join of the stock_df and BB_df on Market_date column.
    result_set = engine.execute('select * from "stock_data_'+symbol+'"')
    #df_join is a dataframe established from result_set
    df_join = pd.DataFrame(result_set)
    df_join.columns = ['Market_date', 'Open Price $', 'High Price $','Low Price $', 'Close Price $', 'Volume','Adjusted Close Price $', '1','2']

    # print(stock_df.values)
    return df_join


def getDataWithIndicator(symbol):
    #result_set is a join of the stock_df and BB_df on Market_date column.
    print('select * from "stock_data_'+symbol+'" inner join "bollinger_bands_'+symbol+'" using ("Market_Date")')
    result_set = engine.execute('select * from "stock_data_'+symbol+'" inner join "bollinger_bands_'+symbol+'" using ("Market_Date")')
    #df_join is a dataframe established from result_set
    df_join = pd.DataFrame(result_set)
    df_join.columns = ['Market_date', 'Open Price $', 'High Price $','Low Price $', 'Close Price $', 'Volume','Adjusted Close Price $','RSI', '20d_MA','Adjusted Close Price $_m', 'bb_bbh','bb_bbl', 'bb_bbhi', 'bb_bbli','bb_bbw', 'bb_bbp', '']

    # print(stock_df.values)
    return df_join

# # Machine Learning
#Linear Regression Machine Learning

def trainModel(stock_df):
    model = LinearRegression()

    X = stock_df[['Open Price $','High Price $','Low Price $','Volume']]
    X.reset_index(drop = True, inplace = True )
    X.to_numpy()

    y = stock_df.pop('Adjusted Close Price $')

    #train and test the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model.fit(X_train, y_train)

    training_score = model.score(X_train, y_train)
    testing_score = model.score(X_test, y_test)

    print("Training ML is working")
    return model

# Close price prediction
def predictPrice(model, Open_price, High_price,  Low_price, Volume):
    X_pred = [Open_price,High_price, Low_price, Volume]
    X_pred = np.array(X_pred).reshape(1,4)
    X_pred
    y_pred = model.predict(X_pred)
    return y_pred

def train_and_predict(stock_df, Open_price, High_price,  Low_price, Volume):
    model = trainModel(stock_df)
    return predictPrice(model, Open_price, High_price,  Low_price, Volume)
    