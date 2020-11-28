#!/usr/bin/env python
# coding: utf-8
"""dashboard

Simple dask web app to help personal portfolio management. Functionality in
portfoliomanager module.

Todo:
    * Portfolio contents pie
    * Portfolio notes visible
"""

#libraries
import pandas as pd
import datetime
    
# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots #not used atm
# Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# import dash_auth
# from username_and_password_pairs import VALID_USERNAME_PASSWORD_PAIRS

# Facebook's library for time series predictions using PyStan
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

import textwrap # for slitting title text

# Jupyter_dash debugging if True
debug = True
if debug:
    from jupyter_dash import JupyterDash #debugging only




### Data loading

import portfoliomanager as pm

portfolio_transactions = {}
portfolio_cash = {}

portfolio_name = 'Own'
filenames = [
    'transactions_export_aot.csv',
    'transactions_export_ost.csv'
]    
portfolio_transactions[portfolio_name], portfolio_cash[portfolio_name] = pm.load_transtactions(filenames)

portfolio_name = 'Managed'
filenames = [
    'transactions_export_apteekki.csv',
]
portfolio_transactions[portfolio_name], portfolio_cash[portfolio_name] = pm.load_transtactions(filenames)


df_inderes_hist = pm.load_inderes_file('company_history_data.csv')

# Now static loading from file, need to implement web scraping here to automize
df_inderes_suosit = pm.load_inderes_file('inderes_osakevertailu_2020_10_21_13_56_56.csv')

# Dictionary of downloadable stocks and their tickers
# Needs moving to external file later
ticker_names_dict = {
    #pankit
    'AKTIA.HE': 'Aktia',
    'NDA-FI.HE': 'Nordea',
    'SAMPO.HE': 'Sampo',
    #sijoitus
    'CAPMAN.HE': 'CapMan',
    #it
    'DIGIA.HE': 'Digia',
    'IFA1V.HE':'Innofactor',
    'TIETO.HE':'TietoEVRY',
    'GOFORE.HE':'Gofore',
    'SIILI.HE':'Siili Solutions',
    'VINCIT.HE':'Vincit',
    'LEADD.HE': 'LeadDesk',
    'QTCOM.HE':'Qt',
    #puollustusteknologia & tietoturva
    'BITTI.HE':'Bittium',
    #pelifirmat
    'REMEDY.HE':'Remedy',
    #terveys
    'PIHLIS.HE':'Pihlajalinna',
    #silmäterveysteknologia
    'REG1V.HE':'Revenio Group',
    #lääke
    'ORNBV.HE':'Orion',
    #sähköyhtiöt
    'FORTUM.HE':'Fortum',
    #syklinen teollisuus
    'RAUTE.HE':'Raute',
    'UPM.HE':'UPM',
    'WRT1V.HE':'Wärtsilä',
    'CGCBV.HE':'Cargotec',
    'GLA1V.HE':'Glaston',
    'ICP1V.HE':'Incap',
    #tasainen tuotto
    'HARVIA.HE':'Harvia',
    #teleyhtiöt
    'TELIA.ST':'Telia company',
    'ELISA.HE':'Elisa',
    'QDVE.DE':'iShares S&P 500 Information Technology Sector UCITS ETF USD (Acc)',
}

dict_timelines = pm.download_stock_timelines(list(ticker_names_dict.keys()))

#needs moving to external file later
index_names_dict = {
    '^OMXH25':'OMXH25',
    '^GSPC':'S&P500',
    '000001.SS':'SSE Composite Index',
    '^N225':'Nikkei 225',
    }

dict_index = pm.download_stock_timelines(list(index_names_dict.keys()))

# Read notes
df_muistiinpanot = pm.read_notes('osakeseuranta_2.xlsx')

# Insider trades
df_sisapiirin_kaupat_per_day = pm.read_insider_trades('sisapiirin_kaupat.csv')

# Length of history in plots
history_days = 365*2

start_date = datetime.datetime.now() - datetime.timedelta(days=history_days)
fig_timeline = pm.make_fig(dict_timelines, dict_index, ticker_names_dict, index_names_dict, df_muistiinpanot, df_sisapiirin_kaupat_per_day, portfolio_transactions['Own'], 10, start_date)
fig_inderes_potential = pm.make_fig_inderes_potential(df_inderes_suosit)
fig_potential = pm.make_fig_potential(ticker_names_dict, dict_timelines, df_inderes_suosit)

# Empty placeholder figures for future prictions done via callback
fig_future = go.Figure()
fig_components = go.Figure()






### App configuration

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

if debug:
    app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
else:
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server # the app.server is the Flask app

# if __name__ != "__main__": #auth doesn't seem to work with jupyterdash debugging, so enable only when run on wsgi
#     auth = dash_auth.BasicAuth(
#         app,
#         VALID_USERNAME_PASSWORD_PAIRS
#     )
# app.config.suppress_callback_exceptions = True

modeBarButtonsToRemove = [
     'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'handleDrag3d', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d',
 'hoverClosestCartesian', 'hoverCompareCartesian',
 'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo',
 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toImage', 'sendDataToCloud', 'toggleSpikelines', 'resetViewMapbo',
]
no_buttons_config = {'modeBarButtonsToRemove': modeBarButtonsToRemove,'displaylogo': False}
config = {'displaylogo': False}

# Styling
app_style = {'font-family': 'Arial'}
dropdown_style = {
    'marginBottom': 0, 'marginTop': 5, 'marginLeft': 0, 'marginRight': 0,
}
label_style = {
    'marginBottom': 0, 'marginTop': 0, 'marginLeft': 75, 'marginRight': 75,
}

def get_dropdown_options(d):
    """Helper to turn dictionary keys into dropdown options into 

    Args:
        d: Dictionary with tickers as keys 

    Returns:
        options: List containing dropdown options
    """
    options = []
    for s in d.keys():
        options.append({'label':s, 'value':s})
    return options

# App layout
app.layout = html.Div(children=[
    html.Div([
        html.Label([
            "Selected stock tickers", 
            dcc.Dropdown(
                id='stock-ticker-input',
                options=get_dropdown_options(ticker_names_dict),
                value=list(ticker_names_dict.keys()),
                multi=True
                #style=dropdown_style
            ),
        ]),
        html.Label([
            "Selected index tickers", 
            dcc.Dropdown(
                id='index-ticker-input',
                options=get_dropdown_options(index_names_dict),
                value=list(index_names_dict.keys()),
                multi=True
                #style=dropdown_style
            ),
        ]),
        html.Button('Draw graphs', id='btn-1'),
        html.Div([
            dcc.Graph(id="fig_timeline", figure=fig_timeline, config=config)
        ]),
        html.Div([
            dcc.Graph(id="fig_inderes_potential", figure=fig_inderes_potential, config=config)
        ]),
        html.Div([
            dcc.Graph(id="fig_potential", figure=fig_potential, config=config)
        ]),
        html.Label([
            "Predicted ticker", 
            dcc.Dropdown(
                id='predict-ticker-input',
                options=get_dropdown_options(ticker_names_dict),
                value=list(ticker_names_dict.keys())[0],
                multi=False
                #style=dropdown_style
            ),
        ]),
        html.Div([
            dcc.Graph(id="fig_future", figure=fig_future, config=config)
        ]),
        html.Div([
            dcc.Graph(id="fig_components", figure=fig_components, config=config)
        ]),
    ], className="row"),
])

# Callbacks
@app.callback(
    Output('fig_timeline', 'figure'),
    [
    Input('btn-1', 'n_clicks'),
    ], 
    [
    State('stock-ticker-input', 'value'),
    State('index-ticker-input', 'value'),
    ],
    prevent_initial_call=True)
def update_timelines(n_clicks, stock_tickers, index_tickers):

    # filtered global var for now, change later to cache or file based storage so adding of completely new tickers stick
    ticker_names_dict_f = { k: ticker_names_dict[k] for k in stock_tickers }
    index_names_dict_f = { k: index_names_dict[k] for k in index_tickers }

    dict_timelines = pm.download_stock_timelines(stock_tickers)
    dict_index = pm.download_stock_timelines(index_tickers)

    fig_timeline = pm.make_fig(dict_timelines, dict_index, ticker_names_dict_f, index_names_dict_f, df_muistiinpanot, df_sisapiirin_kaupat_per_day, portfolio_transactions['Own'], 10, start_date)

    return fig_timeline

@app.callback(
    [
    Output('fig_future', 'figure'),
    Output('fig_components', 'figure'),
    ],
    [
    Input('predict-ticker-input', 'value'),
    ], 
    prevent_initial_call=True)
def update_predictions(ticker):

    df_timeline_with_future, m = pm.predict_future(dict_timelines[ticker])
    fig_future = plot_plotly(m, df_timeline_with_future)
    fig_components = plot_components_plotly(m, df_timeline_with_future)
    return fig_future, fig_components

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8889)