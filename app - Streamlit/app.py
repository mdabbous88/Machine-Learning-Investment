# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from pandas.core.frame import DataFrame
import streamlit as st
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import mlinv

st.sidebar.title("Stock Predict")
# symbol = st.sidebar.text_input("Input Ticker Symbol")
symbols = tuple(pd.read_csv('stocks.csv')["Symbol"].to_list())
symbol = st.sidebar.selectbox(label="Select stock symbol", options=symbols)

def fetchData(symbol):
    return mlinv.runAll(symbol)

button1 = st.sidebar.button("GET DATA")
df = None

if button1:
    df = fetchData(symbol)
    st.write(df)
    fig = go.Figure(data=[go.Candlestick(x=df['Market_date'], open=df['Open Price $'], high=df['High Price $'], low=df['Low Price $'], close=df['Close Price $'])])
    st.plotly_chart(fig)

input_open = st.sidebar.number_input("Open price")
input_high = st.sidebar.number_input("High price")
input_low = st.sidebar.number_input("Low price")
input_vol = st.sidebar.number_input("Volume")

button2 = st.sidebar.button("PREDICT NOW")

if button2:
    prediction = mlinv.train_and_predict(df, input_open, input_high,  input_low, input_vol)
    st.write(prediction)
