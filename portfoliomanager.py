#!/usr/bin/env python
# coding: utf-8
"""portfoliomanager

Module containing functionality for a small dask web app to help
personal portfolio management. A lot of the functionality needs csv
exports from Inderes website. Uses Yahoo API to download stock price
information. Some very experimental time series forecasting
done with fbprophet and skearn.

Todo:
    * Check Wärtsilä transactions, not counted corretly now
    * Timeline figure drawing function needs buttons for flexible timeline
    * Now just a plain module with a lot of functions looping through 
    dictionary of dataframes containing time series data. Better design is 
    most likely to make class containing fucntionality for single ticker dataframe.
    * Web scraping to fetch some of the csv files automatically
"""

# libraries
# from os import path
import pathlib
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 100
pd.options.display.max_rows = 100

import time #to time code

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Financial stuff
# import pandas_datareader as pdr #ei käytössä atm, koska kaikki data YF:llä nyt. Tässä moduulissa kuitenkin sqlite cache option niin täällä siltä varalta jos halutaan käyttää myöhemmin
# import quandl #registered here with api key, but aktia not found fe. aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")
import datetime
# import pyfolio as pf #tää hylätty, koska ei tarvetta riskianalyysille, tarjoaa ison paketin valmiissa graafeissa
# from yahoofinancials import YahooFinancials #2019 alun moduuli, palauttaa kaiken JSON
import yfinance as yf

import requests

# Facebook's library for time series predictions using PyStan
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

# sklearn
import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Basic plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
    
class PortfolioManager:

    def __init__(self):

        def load_transtactions(filename):
            """Loads all portfolio transactions from csv files exported from Inderes website.

            Args:
                filename: csv file
            
            Returns:
                df_trans: Dataframe of transactions.
            """
            df_trans = pd.read_csv(filename, encoding='utf-16 LE', sep='\t',decimal=',',thousands=' ')
            df_trans.reset_index(inplace=True)
            df_trans['Kirjauspäivä'] = pd.to_datetime(df_trans.Kirjauspäivä)

            # nested helper to connect valid ticker information to ticker names found in CSVs
            def get_stock_names(s):
                
                # Map of standard ticker names to various messy Arvopaperi names found in nordnet transactions files. Needs a bit manual updating
                arvopaperit_dict = {
                    'AKTIA.HE' : ['AKTIA'],
                    'BITTI.HE' : ['BITTI'],
                    'Caverion' : ['CAV1V'],
                    'DIGIA.HE' : ['DIGIA'],
                    'GOFORE.HE' : ['GOFORE'],
                    'IFA1V.HE' : ['IFA1V'],
                    'QTCOM.HE' : ['QTCOM'],
                    'Martela' : ['MARAS'],
                    'NDA-FI.HE' : ['NDA FI','NDA1V', 'NDA1V FIKTIV','NDA1VN0109', 'NDA1VU0109'],
                    'NOKIA' : ['NOKIA'],
                    'ORNAV.HE' : ['ORNAV'],
                    'PKC1V.HE' : ['PKC1V'],
                    'Ramirent' : ['RAMI', 'RAMIRENT OSTOTARJOUSOSAKE', 'RAMIRENT OSTOTARJOUSOSAKE/X'],
                    'RAUTE.HE' : ['RAUTE','RTRKS'],
                    'Technopolis' : ['TECHNOPOLIS OSTOTARJOUS HYVÄKSYTTY','TLV1V','TPS1V'],
                    'UPM.HE' : ['UPM'],
                    'WRT1V.HE' : ['WRT1V'],
                    'YIT.HE' : ['YIT'],
                    'QDVE.DE' : ['QDVE'],
                    'QDVE.DE' : ['QDVE'],
                    'TIETO.HE' : ['TIETO'],
                    'REMEDY.HE' : ['REMEDY'],
                    'HEALTH.HE' : ['HEALTH','NIGHTINGALE MERKINTÄSITOUMUS'],
                    'TEM1V.HE' : ['TEM1V'],
                }
                stocks = []
                for v in s:
                    stock_found = False
                    for k in arvopaperit_dict.keys():
                        if v in arvopaperit_dict[k]:
                            stocks.append(k)
                            stock_found = True
                    if stock_found == False:
                        stocks.append(v)
                return stocks
            df_trans['Osake'] = get_stock_names(df_trans.Arvopaperi)

            return df_trans

        def load_inderes_file(filename):
            df_inderes_hist = pd.read_csv(filename, encoding='utf-8', sep=';', header=None, skiprows=[0,1], decimal=',',thousands=' ') #read without headers
            # manually fix 2 row header reading
            infile = open(filename, 'r', encoding='utf-8') 
            firstLine_list = infile.readline()[:-1].split(';')
            secondline_list = infile.readline()[:-2].split(';')
            l = []
            f_pos = 0
            s_pos = 0
            for s in secondline_list:
                if secondline_list[f_pos] == '':
                    l.append(firstLine_list[f_pos])
                else:
                    l.append(firstLine_list[f_pos] + '_' + secondline_list[s_pos])
                s_pos += 1
                if len(firstLine_list) > s_pos and firstLine_list[s_pos] != '':
                    f_pos = s_pos        
            df_inderes_hist.columns = l
            df_inderes_hist.rename(columns={'﻿Yhtiö':'Yhtiö'}, inplace=True) # fix annoying col name
            return df_inderes_hist

        # not needed????
        portfolio_transactions = {}
        portfolio_cash = {}

        portfolio_files = {
            'Own': [
                'transactions_export_aot.csv',
                'transactions_export_ost.csv',
            ],
            'Managed' : [
                'transactions_export_apteekki.csv',
            ],
        }

        self.portfolios = {}
        for pname in portfolio_files.keys():
            dfs = []
            for filename in portfolio_files[pname]:
                dfs.append(load_transtactions(filename))
            self.portfolios[pname] = dfs

        self.df_inderes_hist = load_inderes_file('company_history_data.csv')

        # Now static loading from file, need to implement web scraping here to automize
        file_path = pathlib.Path(__file__).parent
        for f in file_path.iterdir():
            if f.name.startswith('inderes_osakevertailu'):
                l = (f.name)

        self.df_inderes_suosit = load_inderes_file(l)

        def read_notes(filename):
            """Read personal notes from excel file to dataframe.
            
            Args:
                filename (str): Path to excel file containing notes
            
            Returns:
                df: Dataframe containing notes
            """
            # todo: salkkumuistiinpanot loogisempia kuin yksittäisten osakkeiden ja ne ei nyt missään näkyvissä
            df_notes = pd.read_excel(filename)
            return df_notes

        # Read notes
        self.df_muistiinpanot = read_notes('osakeseuranta_2.xlsx')

        def read_insider_trades(filename):
            """Reads insider trade information from csv exported from Inderes website.
            
            Args:
                filename: path of csv file
            
            Returns:
                df_sisapiirin_kaupat_per_day: Dataframe containig daily sums of insider trades
            
            """
            df_sisapiirin_kaupat = pd.read_csv(filename)
            df_sisapiirin_kaupat['Päivämäärä'] = pd.to_datetime(df_sisapiirin_kaupat['Päivämäärä']) #to datetime
            df_sisapiirin_kaupat['Yhteensä'] = df_sisapiirin_kaupat['Yhteensä'].apply(lambda x: x.replace(' ', '').replace('EUR', '')).astype(float)
            df_sisapiirin_kaupat.loc[df_sisapiirin_kaupat.Tyyppi.apply(lambda x: True if x in ['Hankinta'] else False), 'Summa'] = df_sisapiirin_kaupat.loc[df_sisapiirin_kaupat.Tyyppi.apply(lambda x: True if x in ['Hankinta'] else False), 'Yhteensä']
            df_sisapiirin_kaupat.loc[df_sisapiirin_kaupat.Tyyppi.apply(lambda x: True if x in ['Luovutus'] else False), 'Summa'] = df_sisapiirin_kaupat.loc[df_sisapiirin_kaupat.Tyyppi.apply(lambda x: True if x in ['Luovutus'] else False), 'Yhteensä'] * - 1
            def utuple(s): #returns tuple of all found values
                l = s.unique()        
                if len(l) == 1: #don't return tuple with single value
                    l = l[0]
                else:
                    l.sort()
                    l = tuple(l)
                return l
            def flatten(df): #flatten df to get rid of 2-level column names
                df.columns = ['_'.join(tuple(map(str, t))) for t in df.columns.values]
                df.columns = [t.replace('_utuple','') for t in df.columns.values]
                df.reset_index(inplace=True)
                return df
            sisapiirin_kaukat_aggregation = {
                                'Tyyppi' : utuple,
                                'Summa': ['sum'], 
                                }
            df_sisapiirin_kaupat_per_day = flatten(df_sisapiirin_kaupat.groupby(['Yhtiö','Päivämäärä']).agg(sisapiirin_kaukat_aggregation))
            return df_sisapiirin_kaupat_per_day

        # Insider trades
        self.df_sisapiirin_kaupat_per_day = read_insider_trades('sisapiirin_kaupat.csv')

def get_symbol(clear_name):
    """Uses Yahoo API to return ticker name agains clean company name

    Args:
        clear_name (str): Company name found fe. in inderes recommendations export file

    Returns:
        str: ticker name.
    """
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(clear_name)
    result = requests.get(url)
    if result.status_code == 404:
        print('HTML reponse 404: Failed to get ticker name for ' + clear_name )
        return None
    else:
        result_json = result.json()
        for x in result_json['ResultSet']['Result']:
            if x['exch'] == 'HEL':
                return x['symbol']

def download_stock_timelines(tickers):
    """Downloads daily ticker price information since 2015 using Yahoo Finance Python API.
    
    Args:
        tickers (list of str): List of downloaded tickers.
    
    Returns:
        df: Dict using tickers as keys and dataframe of ticker prices over time.
    """
    start = time.time()

    # Timeline code here, but now static, needs to be put to parameters later on
    timeline_start = datetime.datetime(2015, 2, 1) # ~last 5y for now
    timeline_end = datetime.datetime.now()
    full_data_range = pd.date_range(timeline_start - datetime.timedelta(days=1), timeline_end, freq='B') # take -1 from start day to make timeline inclusice
    timeline_start = min(full_data_range) # real min for timeline, counting business days

    dict_timelines = {}

    df_tmp = yf.download(tickers, period='5y', interval='1d') #multi-level column df in case more than 1 ticker
    code_time = time.time() - start
    print('Stock history download time time: %.2f' % code_time)

    # avoid referencing column names here to be robust against yahoo api changes
    index_name = df_tmp.index.name
    df_tmp_m = pd.melt(df_tmp.reset_index(), id_vars=index_name)

    for s in tickers:
        # Future todo: cache system if amount of tickers grows big
        # Implementing own cache a bit tricky cause Adj close changes over whole timeline in case of splits etc.
        # Most likely better to use this: https://pandas-datareader.readthedocs.io/en/latest/cache.html
        if len(tickers) > 1:
            df_tmp2 = df_tmp_m.loc[(df_tmp_m['variable_1'] == s)].set_index([index_name]) # slice indivisual stock out
            df_tmp2 = df_tmp2.pivot(columns='variable_0', values='value') # variable_0 should be lower col level of multi-index df. 'Adj Close' & others.
        else:
            df_tmp2 = df_tmp_m.set_index([index_name]) #whole df is just one stock
            df_tmp2 = df_tmp2.pivot(columns='variable', values='value')

        # Sometimes Yahoo API returns clearly too high numbers for Adj Close, remove these
        f = (df_tmp2['Adj Close'] < (df_tmp2['Adj Close'].mean() * 100))
        df_tmp2 = df_tmp2.loc[f]
        dict_timelines[s] = df_tmp2

        # Missing value filling is disabled for now since we want things to work with weekends not being in the data.
        # dict_timelines[s] = dict_timelines[s].dropna(how='all').reindex(full_data_range) #add missing value rows
        # dict_timelines[s].index.name = 'Date'
        # dict_timelines[s]['Adj Close'] = dict_timelines[s]['Adj Close'].fillna(method='ffill').fillna(method='bfill')  #fill missing closing price, first from back, then from forward (for those with only single value)
        # dict_timelines[s]['Close'] = dict_timelines[s]['Close'].fillna(method='ffill').fillna(method='bfill')  #fill missing closing price, first from back, then from forward (for those with only single value)
    return dict_timelines

def download_ticker_data(tickers):
    """Downloads general information about tickers using Yahoo Finance Python API.
    
    Args:
        tickers (list of str): List of downloaded tickers.
    
    Returns:
        df: Dataframe containing general information about company and some most common KPIs.
    
    """
    dict_pe_forward = {}
    dict_pe_trail = {}
    dict_pb = {}
    dict_bsummary = {}
    for s in tickers:
        ticker = yf.Ticker(s)
        ticker_info_dict = ticker.info #quite slow, bad library
        
        if 'forwardPE' in ticker_info_dict.keys():
            dict_pe_forward[s] = ticker_info_dict['forwardPE'] #osakkeen pe
        else:
            dict_pe_forward[s] = 0
        if 'trailingPE' in ticker_info_dict.keys():
            dict_pe_trail[s] = ticker_info_dict['trailingPE'] #osakkeen pe
        else:
            dict_pe_trail[s] = 0
        if 'priceToBook' in ticker_info_dict.keys():
            dict_pb[s] = ticker_info_dict['priceToBook'] #osakkeen pe
        else:
            dict_pb[s] = 0
        if 'longBusinessSummary' in ticker_info_dict.keys():
            dict_bsummary[s] = ticker_info_dict['longBusinessSummary'] #osakkeen pe
        else:
            dict_bsummary[s] = 0
        print('All ticker data extracted for: ' + s)
    df_stock_stats = pd.DataFrame()
    df_stock_stats['Osake'] = tickers
    df_stock_stats['forwardPE'] = dict_pe_forward.values()
    df_stock_stats['trailingPE'] = dict_pe_trail.values()
    df_stock_stats['priceToBook'] = dict_pb.values()
    df_stock_stats['Business Summary'] = dict_bsummary.values()
    return df_stock_stats

def get_portfolio_summary(transactions, dict_timelines, df_inderes_suosit, ticker_names_dict):
    """Creates summary of portfolio contents based on given list of 
    transactions. Only works with Nordnet style export for now.
    
    Args:
        transactions (list of df): List of datafranes containing transactions read from Nordnet export file.
        dict_timelines (dict of df): Stock price information.
    
    Returns:
        df: Dataframe containig contents of portfolio.
    
    """

    def flatten(df): #flatten df to get rid of 2-level column names
        df.columns = ['_'.join(tuple(map(str, t))) for t in df.columns.values]
        df.columns = [t.replace('_utuple','') for t in df.columns.values]
        df.reset_index(inplace=True)
        return df

    #asetukset
    kauppatapahtumat = ['OSTO','MYYNTI','LUNASTUS AP OTTO','MAKSUTON OSAKEANTI / BONUS ISSUE']
    osinkotapahtumat = ['OSINGON PERUUTUS', 'OSINKO','ENNAKKOPIDÄTYS']
    summary_aggregation = {
                        'Kirjauspäivä': ['max'],
                        }

    df_s = pd.DataFrame()
    cash = 0

    for df_transactions in transactions:

        cash += df_transactions['Saldo'][0] #cash now can be found on last transaction which is first line in file
        df_summary = df_transactions.groupby(['Osake']).agg(summary_aggregation) #only valid month rows
        df_summary = flatten(df_summary)

        kokonaismaara = []
        nykykurssi = []
        sij_paaoma = []
        osingot = []

        for index, row in df_summary.iterrows():
            df_tmp = df_transactions.loc[(df_transactions.Osake == row.Osake) & df_transactions.Tapahtumatyyppi.apply(lambda x: True if x in osinkotapahtumat else False)]
            osingot.append(df_tmp['Summa'].sum())

            df_tmp = df_transactions.loc[(df_transactions.Osake == row.Osake) & df_transactions.Tapahtumatyyppi.apply(lambda x: True if x in kauppatapahtumat else False)]
            sij_paaoma.append(-df_tmp['Summa'].sum())

            df_tmp = df_tmp.loc[(df_tmp.Kirjauspäivä == df_tmp.Kirjauspäivä.max())]
            kokonaismaara.append(df_tmp['Kokonaismäärä'].values[0])

            if row.Osake in dict_timelines.keys(): 
                nykykurssi.append(dict_timelines[row.Osake].dropna(subset=['Close']).tail(1)['Close'].values[0])
            else:
                nykykurssi.append(0)

        df_summary['Kokonaismäärä'] = kokonaismaara
        df_summary['Nykykurssi'] = nykykurssi
        df_summary['Omistuksen arvo'] = df_summary['Kokonaismäärä'] * df_summary['Nykykurssi']
        df_summary['Tuotto'] = df_summary['Omistuksen arvo'] - sij_paaoma + osingot

        if df_s.shape[0] == 0:
            df_s = df_summary
        else:
            df_s = df_s.merge(df_summary, on=['Osake'], how='outer', suffixes=('','_n'))
            # summed cols
            for c in ['Kokonaismäärä','Omistuksen arvo','Tuotto']:
                f = ~(df_s[c + '_n'].isna())
                df_s.loc[f, c] += df_s.loc[f, c + '_n']
                f = (df_s[c].isna())
                df_s.loc[f, c] = df_s.loc[f, c + '_n']
            for c in ['Nykykurssi']:
                f = (df_s[c].isna())
                df_s.loc[f, c] = df_s.loc[f, c + '_n']
                        
        df_s['Yhtiö'] = df_s['Osake'].map(ticker_names_dict)
        df_s = df_s.merge(df_inderes_suosit[['Yhtiö','Tavoitehinta']], on=['Yhtiö'], how='left')
        df_s['Potentiaali'] = ((df_s['Tavoitehinta'] - df_s['Nykykurssi']) / df_s['Nykykurssi']).round(3)

        df_s = df_s[[
            'Osake',
            'Kokonaismäärä',
            'Nykykurssi',
            'Omistuksen arvo',
            'Tuotto',
            'Potentiaali',
        ]]
    return df_s, cash

def make_fig_contents_pie(df_summary, portfolio_cash, portfolio_name):
    """Creates summary of portfolio contents based on given list of 
    transactions. Only works with Nordnet style export for now.
    
    Args:
        df_summary (df): Dataframe containig contents of portfolio.
        portfolio_cash (float): Available cash.
    Returns:
        fig: Pie of portfolio contents.
    
    """
    df_fig = df_summary.loc[(df_summary['Omistuksen arvo'] > 0), ['Osake', 'Omistuksen arvo','Potentiaali']].copy()
    df_fig = df_fig.append({
        'Osake': 'CASH', 
        'Omistuksen arvo': portfolio_cash,
        'Potentiaali': 0,
        }, ignore_index=True)

    fig = px.sunburst(df_fig, values='Omistuksen arvo', path=['Osake'],color='Potentiaali', color_continuous_scale='RdBu', color_continuous_midpoint=0,
    title='Contents of ' + portfolio_name)
    fig.update_layout(
        margin = dict(t=50, l=10, r=10, b=10)
    )
    return fig

def make_fig(dict_timelines, dict_index, ticker_names_dict, index_names_dict, df_notes, df_insider_trades, portfolio_transactions, sma_days, start_date, dict_predictions=None):
    """Draws figure with data traces of all stocks and indices passed as parameters. 
    
    Args:
        dict_timelines: Dictionary of all available stock timeline data. Tickers as keys, Dataframes as values.
        dict_index: Dictionary of all available index timeline data. Tickers as keys, Dataframes as values.
        ticker_names_dict: Dictionary of all plotted stocks. Tickers as keys, Stock names strings as values.
        index_names_dict: Dictionary of all plotted indices. Tickers as keys, Stock names strings as values.
        df_notes: Dataframe of personal notes on stocks. Plotted as points with popups on graphs.
        df_insider_trades: Dataframe of insider trades of stocks. Plotted as points with popups on graphs.
        portfolio_transactions: Dataframe of all personal portfolio transactions. Owned amounts will be plotter on secondary y-axis. 
        sma_days: Int for number of days to use in SMA trace.
        start_date: Start data for x-axis
        dict_predictions: Dictionary of all available futre predictoins of stock timeline data. Tickers as keys, Dataframes as values.
    
    Returns:
        fig: Line graph
    
    """    
    fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
    fig_timeline.update_layout(height=1300) # optimised for 1440p screen now

    for s in ticker_names_dict.keys():
        f = (dict_timelines[s].index > start_date)
        if dict_timelines[s].loc[f].shape[0] > 0:
            df = dict_timelines[s].loc[f].copy()
            scale_value = df['Adj Close'][0]
            df['Scaled Adj Close'] = df['Adj Close'] / scale_value
            
            # stock price
            fig_timeline.add_trace(go.Scatter(x=df.index, y=df['Scaled Adj Close'],
                        mode='lines',
                        line = dict(width=1),
                        name=ticker_names_dict[s]))
            # rolling mean
            fig_timeline.add_trace(go.Scatter(x=df.index, y=df['Scaled Adj Close'].rolling(sma_days).mean(),
                        mode='lines',
                        name=ticker_names_dict[s] + ' SMA ' + str(sma_days)))
            fig_timeline.data[len(fig_timeline.data)-1]["visible"] = 'legendonly' # hide by default
            # future prediction
            if dict_predictions:
                df2 = dict_predictions[s].loc[dict_predictions[s].index > start_date].copy()
                df2['Scaled Adj Close'] = df2['Adj Close'] / scale_value
                fig_timeline.add_trace(go.Scatter(x=df2.index, y=df2['Scaled Adj Close'],
                            mode='lines',
                            name=ticker_names_dict[s] + ' prediction'))
            # notes
            df_notes_s = df_notes.loc[df_notes['Osake'] == s].merge(df, left_on=['Arvio tehty'], right_index=True)
            if df_notes_s.shape[0] > 0:
                fig_timeline.add_trace(go.Scatter(x=df_notes_s['Arvio tehty'], y=df_notes_s['Scaled Adj Close'],
                            mode='markers',
                            name=ticker_names_dict[s] + ' notes'))
            # insider trades
            df_insider_trades_s = df_insider_trades.loc[df_insider_trades['Yhtiö'] == s].merge(df, left_on=['Päivämäärä'], right_index=True)
            if df_insider_trades_s.shape[0] > 0:
                fig_timeline.add_trace(go.Scatter(x=df_insider_trades_s['Päivämäärä'], y=df_insider_trades_s['Scaled Adj Close'],
                            mode='markers',
                            name=ticker_names_dict[s] + ' insider trades'))
            # portfolio transactins
            portfolio_transactions_s = portfolio_transactions.loc[portfolio_transactions['Osake'] == s].merge(df.reset_index(), left_on=['Kirjauspäivä'], right_on=['Date'], how='right') #right_index=True, how='right')
            if portfolio_transactions_s.shape[0] > 0:
                portfolio_transactions_s.sort_values(['Date'], inplace=True)
                # make sure first date has 0 owned stocks for line to draw from there
                portfolio_transactions_s.loc[portfolio_transactions_s.head(1).index, 'Kokonaismäärä'] = portfolio_transactions_s.loc[portfolio_transactions_s.head(1).index, 'Kokonaismäärä'].fillna(0)
                portfolio_transactions_s['Kokonaismäärä'] = portfolio_transactions_s['Kokonaismäärä'].ffill()
                fig_timeline.add_trace(go.Scatter(x=portfolio_transactions_s['Date'], y=portfolio_transactions_s['Kokonaismäärä'],
                            mode='lines',
                            name=ticker_names_dict[s] + ' ownded'), secondary_y=True,)
                fig_timeline.data[len(fig_timeline.data)-1]["visible"] = 'legendonly' # hide by default

    for s in index_names_dict.keys():
        df = dict_index[s].loc[dict_index[s].index > start_date].copy()
        df['Scaled Adj Close'] = df['Adj Close'] / df['Adj Close'][0]
        fig_timeline.add_trace(go.Scatter(x=df.index, y=df['Scaled Adj Close'],
                    mode='lines',
                    line = dict(dash='dot'),
                    name=index_names_dict[s]))
        fig_timeline.data[len(fig_timeline.data)-1]["visible"] = 'legendonly' # hide by default

    return fig_timeline

def make_fig_inderes_potential(df_inderes_suosit):
    """Draws figure of top 100 stocks with most potential found among Inderes recommendations.

    Args:
        df_inderes_suosit (df): Dataframe of of all Inderes recommendations

    Returns:
        fig: Bar graph
    """
    df_tmp = df_inderes_suosit.sort_values(by=['Potentiaali'], ascending=False).reset_index(drop=True).loc[:100] #inderes suosit 100 parasta
    df_tmp['colors'] = df_tmp['Suositus'].map({'Lisää':'lightgreen', 'Vähennä':'orange', 'Osta':'green', 'Myy':'red'})
    fig = px.bar()
    fig.add_trace(go.Bar(x=df_tmp['Yhtiö'], y=df_tmp['Potentiaali'], marker_color=df_tmp['colors'], hovertext=df_tmp['Suositus']
                ))
    fig.update_layout(
        title="Top 100 stocks with most potential found among Inderes recommendations",
        yaxis_title="Potential (%)",
    )
    return fig

def make_fig_potential(ticker_names_dict, dict_timelines, df_inderes_suosit):
    """
    Draws figure of latest stock prices vs latest price recommendations from Inderes.

    Args:
        ticker_names_dict: Dictionary of all plotted stocks. Tickers as keys, Stock names strings as values.
        dict_timelines: Dictionary of all available stock timeline data. Tickers as keys, Dataframes as values.
        df_inderes_suosit: Dataframe of of all Inderes recommendations

    Returns:
        fig: Bar graph
    """
    indices, closes, targets, suositukset = [], [], [], []
    
    for s in ticker_names_dict:

        # if recommendation is found
        f = (df_inderes_suosit['Yhtiö'] == ticker_names_dict[s])
        if df_inderes_suosit.loc[f].shape[0] > 0:
            # only plot if recommendation price is in EUR. For 2 companies this is in SEK.
            if df_inderes_suosit.loc[f, 'Valuuttakurssi'].values[0] == 'EUR': 
                indices.append(ticker_names_dict[s])
                closes.append(dict_timelines[s].tail(1)['Close'].values[0])
                targets.append(df_inderes_suosit.loc[df_inderes_suosit['Yhtiö'] == ticker_names_dict[s], 'Tavoitehinta'].values[0]) # this should always produce one row anyway
                suositukset.append(df_inderes_suosit.loc[df_inderes_suosit['Yhtiö'] == ticker_names_dict[s], 'Suositus'].values[0]) # this should always produce one row anyway

    df_tmp = pd.DataFrame(index=indices, data={'Close':closes,'Target':targets, 'Suositus':suositukset})
    df_tmp['Potential'] = (df_tmp['Target'] - df_tmp['Close']) * 100 / df_tmp['Close'] # in %
    df_tmp = df_tmp.sort_values(by=['Potential'], ascending=False).reset_index()
    df_tmp['colors'] = df_tmp['Suositus'].map({'Lisää':'lightgreen', 'Vähennä':'orange', 'Osta':'green', 'Myy':'red'})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_tmp['index'], y=df_tmp['Potential'], marker_color=df_tmp['colors'], hovertext=df_tmp['Suositus']
                ))

    plotted_c = len(fig.data[0]['x'])
    all_c = df_inderes_suosit.shape[0]
    
    fig.update_layout(
        title="Latest price vs Inderes recommendation (" + str(plotted_c) + "/" + str(all_c) + " companies)",
        yaxis_title="Potential (%)",
    )
    return fig

def predict_future(df_timeline, days=365):
    """Uses fbprophet to produce forecasted results and fitted prediction model for a stock

    Args:
        df_timeline: Dataframe of daily prices for a selected stock. Prediction is fitter to whole Dataframe.
        days: Number of days into the future we are prediction

    Returns:
        df_timeline_with_future: Dataframe of daily prices for a selected stock extending into the future with predictions. Column names are in fbprophet compatible format.
        m: Fitted prediction model for a stock.
    """
    df_timeline_p = df_timeline.reset_index().rename(columns={'Date':'ds','Adj Close':'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df_timeline_p)
    df_timeline_with_future = m.make_future_dataframe(periods=days)
    df_timeline_with_future = m.predict(df_timeline_with_future)
    return df_timeline_with_future, m

def model_selection(df_timeline):
    """Work in progress. Function for testing reg models in individual stock timeline.

    Args:
        df_timeline: Dataframe of daily prices for a selected stock. Prediction is fitter to whole Dataframe.

    Returns:
        fig: Matplotlib boxplot of mean cross validation score with standard deviation.
        models: List of all tested regression models. Elements are tuples with name of model and model.

    """

    # Now really simple features used for testing models. Yesteday's price + Diff between yesterday and day before.
    # Needs longer seasonality trends and most likely index values as additional features.
    df_timeline_f = df_timeline[['Adj Close']].copy()
    df_timeline_f.loc[:,'Yesterday'] = df_timeline_f.loc[:,'Adj Close'].shift()
    df_timeline_f.loc[:,'Yesterday_Diff'] = df_timeline_f.loc[:,'Yesterday'].diff()
    df_timeline_f = df_timeline_f.dropna() # check later where nas coming

    time_split_date = df_timeline_f.tail(21)[:1].index[0] # last 21 days as test data.
    X_train = df_timeline_f.loc[df_timeline_f.index < time_split_date].drop(['Adj Close'], axis = 1)
    y_train = df_timeline_f.loc[df_timeline_f.index < time_split_date, 'Adj Close']
    X_test = df_timeline_f.loc[df_timeline_f.index >= time_split_date].drop(['Adj Close'], axis = 1)
    y_test = df_timeline_f.loc[df_timeline_f.index >= time_split_date, 'Adj Close']

    models = []
    models.append(('LR', LinearRegression()))
    models.append(('NN', MLPRegressor(solver = 'lbfgs'))) # nn
    models.append(('KNN', KNeighborsRegressor())) 
    models.append(('RF', RandomForestRegressor(n_estimators = 10)))
    models.append(('SVR', SVR(gamma='auto'))) # kernel = linear
    
    # Train set cross val scores
    results = []
    names = []
    for name, model in models:
        tscv = TimeSeriesSplit() # default 5 split ok for us
        cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        
    # Boxplot
    fig = plt.boxplot(results, labels=names)

    # to be continued here with automatic test set comparison, maybe autoselection
    return fig, models, results, df_timeline_f

def predict_using(reg, df_timeline):
    """Work in progress. Uses provided model to predictict future Adj Close price based on provided df_timeline.

    Args:
        model: Regression model. Will be fitted in the provided data.
        df_timeline: Dataframe of daily prices for a selected stock. Prediction is fitter to whole Dataframe.
    Returns:
        df_predictions: Dataframe of future daily prices for a selected stock.
        
    """
    # fit the model to whole data
    df_timeline_f = df_timeline[['Adj Close']]
    df_timeline_f.loc[:,'Yesterday'] = df_timeline_f.loc[:,'Adj Close'].shift()
    df_timeline_f.loc[:,'Yesterday_Diff'] = df_timeline_f.loc[:,'Yesterday'].diff()
    df_timeline_f = df_timeline_f.dropna() # check later where nas coming

    X = df_timeline_f.drop(columns=['Adj Close'])
    Y = df_timeline_f[['Adj Close']]
    reg.fit(X, Y)

    pred_first_date = df_timeline_f.tail(1)[:1].index[0] + datetime.timedelta(days=1)
    pred_last_date = pred_first_date + datetime.timedelta(days=63) # about three months into the future

    X_yesterday = df_timeline_f.tail(1).drop(columns=['Adj Close']) # df
    Y_yesterday = df_timeline_f.tail(1)[['Adj Close']].values[0][0] # float
    Y_before_yesterday = df_timeline_f.tail(2)[['Adj Close']].values[0][0] # float

    pred_date = pred_first_date
    delta = datetime.timedelta(days=1)
    
    df_predictions = pd.DataFrame()
    
    # slow append loop, but will do for now
    while pred_date <= pred_last_date:
        Y_today = reg.predict(X_yesterday)[0][0]  # float
        df_t = pd.DataFrame(index=[pred_date], data={'Adj Close' : Y_today, 'Yesterday' : Y_yesterday, 'Yesterday_Diff' : Y_yesterday - Y_before_yesterday})
        df_predictions = df_predictions.append(df_t)

        pred_date += delta
        X_yesterday = df_t.drop(columns=['Adj Close'])
        Y_before_yesterday = Y_yesterday
        Y_yesterday = Y_today
    return df_predictions