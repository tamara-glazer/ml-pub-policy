'''
DESCRIPTION

Author: Tammy Glazer
'''

import numby as np
import pandas as pd
from sodapy import Socrata
import seaborn as sns
import matplotlib.pyplot as plt

URL = 'data.cityofchicago.org'
ENDPOINT = '6zsd-86xi'
TOKEN = 'YNQSgf5W1B65zdCeAX4nY9CXl'


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


def create_groups(df, level_1, level_2):

    crimes_by_type = df.groupby([level_1, level_2], as_index=False).agg\
                                ({'case_number': 'count'}).pivot\
                                (index=level_2, columns=level_1)\
                                ['case_number'].fillna(0).astype(int)
    return groups


def summarize_crimes_by_type(df):

    crimes_by_type = create_groups(df, 'year', 'primary_type')
    crimes_by_type.index.name = None
    crimes_by_type['total'] = crimes_by_type['2017'] + crimes_by_type['2018']
    crimes_by_type['percent_change'] = crimes_by_type.pct_change\
                                       (axis='columns')['2018'].round(2)


    #return a table

def create_months(df):

    df['date'] = df.date.astype('datetime64[M]')
    df['month'] = df.date.dt.to_period('M')

    return df

def create_crime_heatmap(df):

    df_with_months = create_months(df)
    crimes_by_month = create_groups(df_with_months, 'month', 'primary_type')
    crimes_by_month.index.name = None
    sns.set(font_scale=0.5)
    heatmap = sns.heatmap(crimes_by_month,
                          cmap='Blues',
                          annot=True,
                          annot_kws={'size':8},
                          cbar=False,
                          linewidths=0.5,
                          mask=(crimes_by_month == 0))
    heatmap.set_title('Number of Crimes by Month, 2017-2018')
    plt.show(heatmap)

def summarize_crimes_by_month(df):

    df_with_months = create_months(df)
    df_with_months.groupby(['month'], as_index=False).agg({'case_number':
                                                          'count'})

def summarize_crimes_by_community_area(df):

    df.groupby(['community_area'], as_index=False).agg({'case_number':
               'count', 'arrest': 'sum'})
    #.sort_values(by=['case_number'], ascending=False)
    #ensure that the arrests are calculating correctly
    #more heatmaps by location?

























