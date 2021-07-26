# Group13

## Project purpose

The project purpose is to determine if there is any correlation between stock price movement and technical analysis. Technical analysis uses technical indicators which are mathmatical calculations that are used by traders to predict the future price of a security. The technical indicators that will be used in the study are the Simple Moving Average (SMA), Relative Strength Index (RSI) and Bollinger Bands (BB). The dataset is scraped from Yahoo Finance API based on the stock symbol the user enters. A machine learning model will be created using Close price, Open price, High price, Low price, Adjusted Close price, Volume, SMA, RSI and BB as inputs and the output will be adjusted close price of the security.

## Simple Moving Average
The Simple Moving Average (SMA) is a simple and most commonly used in technical analysis because it helps smooth out the price data over a specified time frame by creating an updated average price.

## Bollinger Bands
A Bollinger Bands is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price, but which can be adjusted to user preferences.

## Relative Strength Index
The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.

<<<<<<< HEAD
## Application Process Flow Diagram

![PFD diagram](Resources/PFD.png)
=======
##Application Process Flow Diagram

![PFD_diagram](https://github.com/mdabbous88/Group13/blob/Ali/Resources/PFD.png)
>>>>>>> 8992fd54d5b3e7de4719eb46d604482382cfb796

## ML Mockup

For the Machine Learning we will use the **sklearn.linear_model.LinearRegression** to predict the **Adjusted Close Price**

![ML Mockup](Resources/ML.png)

## Database Schema

![Database Schema](Resources/PostgreSQL.png)

