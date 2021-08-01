# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from numpy import empty
from pandas.core.frame import DataFrame
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import mlinv

st.header("ML INVESTMENT")
st.image("stocks.jpg")
st.subheader("Using Linear Regression ML")
st.write("You can also try [Classification ML](https://stock-analyse.herokuapp.com)")
st.write("---")
st.sidebar.title("Stock Symbols")
# symbol = st.sidebar.text_input("Input Ticker Symbol")
symbols = tuple(pd.read_csv('stocks.csv')["Symbol"].to_list())
symbol = st.sidebar.selectbox(label="Select stock symbol", options=symbols)


def fetchData(symbol):
    return mlinv.runAll(symbol)

if symbol:
    df = mlinv.runAll(symbol)
    st.write(df)
    fig = go.Figure(data=[go.Candlestick(x=df['Market_date'], open=df['Open Price $'], high=df['High Price $'], low=df['Low Price $'], close=df['Close Price $'])])
    st.plotly_chart(fig)

with st.form('my_form'):
    input_open = st.number_input("Open price")
    input_high = st.number_input("High price")
    input_low = st.number_input("Low price")
    input_vol = st.number_input("Volume")
    predict_button = st.form_submit_button("PREDICT NOW")

    if predict_button:
        prediction = mlinv.train_and_predict(df, input_open, input_high,  input_low, input_vol)
        st.write(prediction)
