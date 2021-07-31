# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import mlinv
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def generateFig(Symbol='AAPL'):
    # 1. get data from database based on Symbol
    # 2. make it as a datafrome
    # 3. modify the column name as the go Figure
    # df = pd.read_csv(Symbol+'.csv')
    
    df = mlinv.runAll(Symbol)
    #df = mlinv.getDataWithIndicator(Symbol)
    print(df.columns)

    fig = go.Figure(data=[go.Candlestick(x=df['Market_date'], open=df['Open Price $'], high=df['High Price $'], low=df['Low Price $'], close=df['Close Price $'])])
    print('fig generated')
    return fig

def generateDropdownList():
    df = pd.read_csv('stocks.csv')
    droptown_items = []
    for index, row in df.iterrows():
        droptown_items.append({'label': row['Name'], 'value': row['Symbol']})
    return droptown_items

# GENERATE HTML
# <div><h1></h1><dcc/><dcc/><div></div></div>

app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        dcc.Dropdown(
            id='symbol-input',
            options=generateDropdownList(),
            value='MMM'
        ),
        dcc.Graph(id='cs-graphic'),
        dcc.Input(id="symbol", type="text", placeholder="Stock", style={'marginRight':'10px'}),
        dcc.Input(id="input_open", type="text", placeholder="Open Price", style={'marginRight':'10px'}),
        dcc.Input(id="input_high", type="text", placeholder="High Price", style={'marginRight':'10px'}),
        dcc.Input(id="input_low", type="text", placeholder="Low Price", style={'marginRight':'10px'}),
        dcc.Input(id="input_vol", type="text", placeholder="Volume", debounce=True),
                html.Div(id="output"),
        

        html.Div(id='my-output')
    ])

# Read document to know how to have multi output in annotation and return vavlue
# Output(component_id='my-output', component_property='children'),
@app.callback(
    Output('cs-graphic', 'figure'),
    Input(component_id='symbol-input', component_property='value')
)
def update_output_div(input_value):
    # logic to change the chart
    return generateFig(input_value)

@app.callback(
    Output("output", "children"),
    Input("symbol", "value"),
    Input("input_open", "value"),
    Input("input_high", "value"),
    Input("input_low", "value"),
    Input("input_vol", "value")
)
def update_output(input_value, input1, input2, input3, input4):
    df = generateFig(input_value)
    prediction = mlinv.predictPrice(df,input1, input2, input3,  input4)
    return u'Open Price {} and High Price {} and Low Price {} and Volume {} and the prediction for today closing price is {}'.format(input1, input2, input3, input4, prediction)

if __name__ == '__main__':
    app.run_server(debug=True)