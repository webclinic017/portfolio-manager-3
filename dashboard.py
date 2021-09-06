#!/usr/bin/env python
# coding: utf-8

"""dashboard

Simple dash web app to help personal portfolio management. Functionality in
portfoliomanager module.

Todo:
    * GUI to add tickers from Yahoo to now premade list 
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
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# import dash_auth
# from username_and_password_pairs import VALID_USERNAME_PASSWORD_PAIRS

# Facebook's library for time series predictions using PyStan
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

import textwrap # for slitting long text


### Data loading

import portfoliomanager as portfoliomanager
pm = portfoliomanager.PortfolioManager()

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

# append ticker_names_dict with companies found in inderes recommendations
for c in pm.df_inderes_suosit['Yhtiö'].tolist():
    symbol = portfoliomanager.get_symbol(c)
    if symbol is not None:
        ticker_names_dict[symbol] = c


#needs moving to external file later
index_names_dict = {
    '^OMXH25':'OMXH25',
    '^GSPC':'S&P500',
    '000001.SS':'SSE Composite Index',
    '^N225':'Nikkei 225',
    }

# Length of history in plots
history_days = round(365*0.25) # about 3m
start_date = datetime.datetime.now() - datetime.timedelta(days=history_days)

### App configuration

external_stylesheets = [dbc.themes.MATERIA]
# external_stylesheets=[dbc.themes.GRID] # grids, but no altering to default theme

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server # the app.server is the Flask app

# if __name__ != "__main__": #auth doesn't seem to work with jupyterdash debugging, so enable only when run on wsgi
#     auth = dash_auth.BasicAuth(
#         app,
#         VALID_USERNAME_PASSWORD_PAIRS
#     )

# modeBarButtonsToRemove = [
#      'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
#  'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'handleDrag3d', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d',
#  'hoverClosestCartesian', 'hoverCompareCartesian',
#  'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo',
#  'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'toImage', 'sendDataToCloud', 'toggleSpikelines', 'resetViewMapbo',
# ]
# no_buttons_config = {'modeBarButtonsToRemove': modeBarButtonsToRemove,'displaylogo': False}

config = {'displaylogo': False}

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
timeline_and_selections = dbc.Row(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label([
                            "Selected stock tickers", 
                            dcc.Dropdown(
                                id='stock-ticker-input',
                                options=get_dropdown_options(ticker_names_dict),
                                value=list(ticker_names_dict.keys()),
                                multi=True
                            ),
                        ]),
                    ], width=6,
                ),
                dbc.Col(
                    [
                        html.Label([
                            "Selected index tickers", 
                            dcc.Dropdown(
                                id='index-ticker-input',
                                options=get_dropdown_options(index_names_dict),
                                value=list(index_names_dict.keys()),
                                multi=True
                            ),
                        ]),
                    ], width=6,
                ),
            ],
            # style = {'background-color' : 'blue'}, #align debugging
        ),
        dbc.Col(
            [
                dbc.Row(
                    dbc.Button('Draw graphs', id='btn-1', style = {'margin-bottom': '2px'}, size='sm'),
                    #style = {'background-color' : 'red'}, #align debugging
                    justify = 'center',
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(id="fig_timeline", config=config),
                        #style = {'background-color' : 'black'}, #align debugging
                    ),
                 ),
            ],
        ),
    ],
    # style = {'background-color' : 'coral'}, #align debugging
)

potential_figure = dbc.Row(
    [
        # html.Div([
        #     dcc.Graph(id="fig_inderes_potential", figure=fig_inderes_potential, config=config)
        # ]),
        dbc.Col([
            dcc.Graph(id="fig_potential", config=config)
        ]),
    ]
)

prediction_figures = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Row(
                    [
                        html.Label([
                            "Predicted ticker", 
                            dcc.Dropdown(
                                id='predict-ticker-input',
                                options=get_dropdown_options(ticker_names_dict),
                                value=list(ticker_names_dict.keys())[0],
                                multi=False,
                                style={'width':'125%'}
                            )
                        ]),
                    ],
                    justify = 'center',
                ),                
                dbc.Row([
                    dcc.Graph(id="fig_future", config=config)

                ]),
               dbc.Row([
                    dcc.Graph(id="fig_components", config=config)
                ]),
            ]
        ),
    ]
)

portfolio_pies = dbc.Row(
    [
        dbc.Col([
            dcc.Graph(id="fig_pie1", config=config)
        ],  width=6,),
        dbc.Col([
            dcc.Graph(id="fig_pie2", config=config)
        ], width=6,),
    ]
)

app.layout = html.Div(
    [
        dbc.Container(
            [
                timeline_and_selections,
                potential_figure,
                portfolio_pies,
                prediction_figures,
            ],
            fluid=True,
        ),
    ], 
    className="dash-bootstrap",
)


# Callbacks
@app.callback(
    Output('fig_timeline', 'figure'),
    Output('fig_potential', 'figure'),
    Output('fig_pie1', 'figure'),
    Output('fig_pie2', 'figure'),

    Input('btn-1', 'n_clicks'),
    State('stock-ticker-input', 'value'),
    State('index-ticker-input', 'value'),
    prevent_initial_call=True)
def update_timelines(n_clicks, stock_tickers, index_tickers):

    # filtered global var for now, change later to cache or file based storage so adding of completely new tickers stick
    ticker_names_dict_f = { k: ticker_names_dict[k] for k in stock_tickers }
    index_names_dict_f = { k: index_names_dict[k] for k in index_tickers }

    dict_timelines = portfoliomanager.download_stock_timelines(stock_tickers)
    dict_index = portfoliomanager.download_stock_timelines(index_tickers)

    fig_timeline = portfoliomanager.make_fig(dict_timelines, dict_index, ticker_names_dict, index_names_dict, pm.df_muistiinpanot, pm.df_sisapiirin_kaupat_per_day, pm.portfolios['Own'][0], 10, start_date) # little workaround with transactions now

    fig_potential = portfoliomanager.make_fig_potential(ticker_names_dict, dict_timelines, pm.df_inderes_suosit)

    # Summary of portfolio contents
    dict_summary = {}
    portfolio_cash = {}
    for p in pm.portfolios:
        df_s, cash = portfoliomanager.get_portfolio_summary(pm.portfolios[p], dict_timelines, pm.df_inderes_suosit, ticker_names_dict)
        dict_summary[p] = df_s
        portfolio_cash[p] = cash

    # Figures of portfolio contents.
    dict_pie_figs = {}
    for p in dict_summary:
        dict_pie_figs[p] = portfoliomanager.make_fig_contents_pie(dict_summary[p], portfolio_cash[p], p)

    return fig_timeline, fig_potential, dict_pie_figs['Own'], dict_pie_figs['Managed']

@app.callback(
    Output('fig_future', 'figure'),
    Output('fig_components', 'figure'),
    Input('predict-ticker-input', 'value'),
    prevent_initial_call=True)
def update_predictions(ticker):

    dict_timelines = portfoliomanager.download_stock_timelines([ticker])

    df_timeline_with_future, m = portfoliomanager.predict_future(dict_timelines[ticker])
    fig_future = plot_plotly(m, df_timeline_with_future)
    fig_components = plot_components_plotly(m, df_timeline_with_future)
    return fig_future, fig_components

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8889)