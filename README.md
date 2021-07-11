# Group13

Purpose and reason: 

## Project Goal 

The project goal is to determine if there is any correlation between stock price movement and technical analysis. Technical analysis uses technical indicators which are mathmatical calculations that are used by traders to predict the future price of a security. The technical indicators that will be used in the study are the Simple Moving Average (SMA), Relative Strength Index (RSI) and Bollinger Bands (BB). The datasets used in the study will be for Amazon stock that is scalped using Yahoo Finance API. A machine learning model will be created using Close price, Open price, High price, Low price, Adjusted Close price, Volume, SMA, RSI and BB as inputs and the output will be a recommendation on wheather to buy, sell or hold the security.

Simple Moving Average

The Simple Moving Average (SMA) is a simple and most commonly used in technical analysis because it helps smooth out the price data over a specified time frame by creating an updated average price.

## Application Process Flow Diagram

![Process Flow Diagram](Resources/PFD.png)

## ML Mockup

For the Machine Learning (ML) we will use the **sklearn.linear_model.LinearRegression** to predict the **Adjusted Close Price**

![ML Mockup](Resources/ML.png)
