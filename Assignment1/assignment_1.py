'''
DESCRIPTION

Author: Tammy Glazer
'''

import numpy as np
import pandas as pd
from sodapy import Socrata
import seaborn as sns
import matplotlib.pyplot as plt
import xlsxwriter

URL = 'data.cityofchicago.org'
ENDPOINT = '6zsd-86xi'
TOKEN = 'YNQSgf5W1B65zdCeAX4nY9CXl'
CSV = 'crimes_2017_to_2018'


def load_data(url=URL, token=TOKEN, endpoint=ENDPOINT):
    '''
    Uses the Socrata Open Data API (SODA) to retrieve data from the Chicago
    Data Portal on crimes reported between 2017 and 2018 using a unique,
    public app token. Converts JSON retrieved from API to a list of
    dictonaries and then to a pandas DataFrame.

    Inputs:
        url (str): url for the City of Chicago Open Data Portal
        token (str): public app token created for "tglazer_assignment1"
        endpoint (str): endpoint details to direct the API

    Output:
        df (dataframe): a pandas dataframe containing data for 2017 and 2018
    '''
    client = Socrata(URL, TOKEN)
    results = client.get(ENDPOINT, where='year >= 2017 and year <=2018')
    df = pd.DataFrame.from_records(results)

    return df

def read_csv(input_file=CSV):
    '''
    INSERT DOCSTRINg
    '''

    csv = pd.read_csv(CSV)
    return csv


def create_groups(df, level_1, level_2):
    '''
    INSERT DOCSTRING
    '''

    groups = df.groupby([level_1, level_2], as_index=False).agg({'Case Number':
                        'count'}).pivot(index=level_2, columns=level_1)\
                        ['Case Number'].fillna(0).astype(int)
    return groups


def summarize_crimes_by_type(df, output_file):
    '''
    INSERT DOCSTRING
    '''

    crimes_by_type = create_groups(df, 'Year', 'Primary Type')
    crimes_by_type.index.name = None
    crimes_by_type['Total'] = crimes_by_type[2017] + crimes_by_type[2018]
    crimes_by_type['Percent Change'] = crimes_by_type.pct_change\
                                       (axis='columns')[2018].round(2)
    crimes_by_type.to_excel(output_file)


def create_months(df):
    '''
    INSERT DOCSTRING
    '''

    df['Date'] = df.Date.astype('datetime64[M]')
    df['Month'] = df.Date.dt.to_period('M')

    return df


def create_crime_heatmap(df):
    '''
    INSERT DOCSTRING
    '''

    df_with_months = create_months(df)
    crimes_by_month = create_groups(df_with_months, 'Month', 'Primary Type')
    crimes_by_month.index.name = None
    crimes_by_month.sortlevel(level=0, ascending=False, inplace=True)
    sns.set(font_scale=0.5)
    heatmap = sns.heatmap(crimes_by_month,
                          cmap='Blues',
                          annot=True,
                          annot_kws={'size': 5},
                          fmt='g',
                          cbar=False,
                          linewidths=0.5,
                          mask=(crimes_by_month == 0))
    heatmap.set_title('Number of Crimes by Month, 2017-2018')
    plt.show(heatmap)
    # should sort

def summarize_crimes_by_month(df, output_file):
    '''
    ADD DOCTSRING
    '''

    df_with_months = create_months(df)
    df_with_months.groupby(['Month'], as_index=False).agg({'Case Number':
                                                          'count'})
    df_with_months.to_excel(output_file)


def summarize_crimes_by_community_area(df, output_file):
    '''
    ADD DOCSTRING
    '''

    df.groupby(['Community Area'], as_index=False).agg({'Case Number':
               'count', 'Arrest': 'sum'}).astype(int).sort_values\
               ('Community Area')
    df.to_excel(output_file)


def summarize_arrests_over_time(df):
    '''
    ADD DOCSTRING
    '''

    arrests_over_time = create_groups(df, 'Arrest', 'Date')
    arrests_over_time.sort_index().cumsum().plot()
    sns.set(font_scale=0.5)
    arrests_over_time.set_title('Number of Reported Crimes by Arrest Status, 2017-2018')
    plt.show()
    #Is this working right?

    # more heatmaps by location???
    # format the tables and put in a document

def provide_summary_stats(df):
    pass

def more_community_area(df):
    pass





























