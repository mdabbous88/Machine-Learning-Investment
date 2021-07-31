import streamlit as st
import pandas as pd
import sklearn, ta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go


days = 3    # how many day to get this profit

st.sidebar.title("Stock Predict")
symbol = st.sidebar.text_input("Input Ticker Symbol")
rate = st.sidebar.slider("Expected Profit rate", min_value=1.01, max_value=1.03)
st.sidebar.write("The prediction is based on close prices in the next few days, if the average of those price great than current day's close price recommend buying.")

def genData(symbol):
    
    
    df = yf.Ticker(symbol).history(period="max")
    
    df["Recommend"]=0
    for i in range(len(df) - days):
        if (df.iloc[i+1:i+1+days,3].mean() > df.iat[i,3]*rate):
            df.iat[i,7]=1
    ta.add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df = df.iloc[37:,:]
    df.fillna(0, inplace=True)
    return df

if symbol:
    ## prepare data
    df = genData(symbol=symbol)
    st.write("Line Chart for Close Price:")
    st.line_chart(df["Close"])
    st.write("Last Five Days Data of", symbol.upper())
    st.write(df.tail())
    X ,y = df.iloc[:-days,8:], df.iloc[:-days,7]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    ## Training Model
    clf = ExtraTreesClassifier()
    clf.fit(X_train_scaled, y_train)
    score = clf.score(X_test_scaled, y_test)
    #st.write("The prediction is based on close prices in the next few days, if the average of those price great than current day's close price recommend buying.")
    #st.write("Score:", score)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Score"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    st.plotly_chart(fig)



    ## Last 100 days predictions
    new_data = X_scaler.transform(df.iloc[-100-days:-days,8:])
    predic_new = clf.predict(new_data)

    st.write("Test Last 100 Days Preditions confusion matrix:")

    predic_df = pd.DataFrame({"Predict": predic_new, "Actul": y[-100:]})
    cm = sklearn.metrics.confusion_matrix(predic_df["Actul"], predic_df["Predict"])
    st.write(cm)
    st.write(f"The app made {cm[1][1]} correct buying predictions and {cm[0][1]} wrong buying predictions for last 100 days")

    ## Last 3day prediction
    last_X = X_scaler.transform(df.iloc[-3:,8:])
    last_predict = clf.predict(last_X)
    last_recommend = bool(last_predict[-1])

    st.write("Last Day prediction:", last_recommend)
    
