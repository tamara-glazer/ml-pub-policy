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
import requests
from uszipcode import SearchEngine
from pylab import savefig


URL_2017 = 'https://data.cityofchicago.org/resource/' \
           '6zsd-86xi.json?year=2017&$limit=600000'
URL_2018 = 'https://data.cityofchicago.org/resource/' \
           '6zsd-86xi.json?year=2018&$limit=600000'
ACS = 'https://api.census.gov/data/2017/acs/acs5/' \
'?get=B01003_001E,B02001_003E,B15003_017E,B08121_001E,B07012_002E&for=zip \
code tabulation area:*'


def load_one_year(url):
    '''
    Uses an HTTP request library to retrieve data from the Chicago Data Portal
    on crimes reported for a single year. Converts a request object to a JSON,
    and finally to a pandast DataFrame.

    Inputs:
        url (str): url from the City of Chicago Open Data Portal (single year)

    Output:
        df (dataframe): a pandas dataframe containing data for a single year
    '''
    request = requests.get(url)
    json = request.json()
    df = pd.DataFrame(json)

    return df


def create_groups(df, x_level, y_level):
    '''
    Groups and collapses a dataframe to display the number of cases by any two
    variables. The variable designated as y_level will appear along the y-axis
    (rows) and the variable designated as x_level will appear along the x-axis
    (columns).

    Inputs:
        df (dataframe): the dataframe to collapse
        x_level (str): name of the variable that will appear in columns
        y_level (str): name of the variable that will appear in rows

    Output:
        groups (dataframe): a summary table presented as a DataFrame
    '''
    groups = df.groupby([x_level, y_level], as_index=False).agg({'case_number':
                        'count'}).pivot(index=y_level, columns=x_level)\
                        ['case_number'].fillna(0).astype(int)
    groups.index.name = None

    return groups


def summarize_crimes_by_type(df, output_file):
    '''
    Creates a summary table of total number of reported cases by type (rows)
    as well as by year (columns). Includes metrics for total and percent
    change between 2017 and 2018.

    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''

    crimes_by_type = create_groups(df, 'year', 'primary_type')
    crimes_by_type['total'] = crimes_by_type['2017'] + crimes_by_type['2018']
    crimes_by_type['percent_change'] = crimes_by_type.pct_change\
                                       (axis='columns')['2018'].round(2)
    crimes_by_type.to_excel(output_file)


def create_crime_heatmap(df, output_file):
    '''
    Creates a heatmap displaying number of reported cases by type (rows)
    as well as by month/year (columns).

    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''

    crimes_by_month = create_groups(df, 'month', 'primary_type')
    crimes_by_month.sort_values('2017-01', ascending=False, inplace=True)
    heatmap = sns.heatmap(crimes_by_month,
                          cmap='Blues',
                          annot=True,
                          annot_kws={'size': 5},
                          fmt='g',
                          cbar=False,
                          linewidths=0.5,
                          mask=(crimes_by_month == 0))
    heatmap.set_title('Number of Crimes by Month, 2017-2018')
    figure = heatmap.get_figure()
    figure.savefig('test.png', dpi=400)


def summarize_crimes_by_month(df, output_file):
    '''
    Creates a summary table of total number of reported cases by month.

    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''
    monthly = df.groupby(['month'], as_index=False).agg({'case_number':
                                                         'count'})
    monthly.rename(columns={'case_number': 'number_of_cases'})
    monthly.to_excel(output_file, index=False)


def summarize_crimes_by_neighborhood(df, output_file):
    '''
    Creates a summary table of total number of cases and arrests reported by
    neighborhood in 2017-2018. The table is sorted by total number of arrests.
    
    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''

    district = df.groupby(['neighborhood'], as_index=False).agg({'case_number':
                          'count', 'arrest': 'sum'})
    district['arrest'] = district['arrest'].astype(int)
    district.sort_values('arrest', ascending=False, inplace=True)
    district.to_excel(output_file, index=False)


def summarize_arrests_over_time(df, output_file):
    '''
    Produces a line plot of the running total of number of reported cases,
    with one line representing cases that end in arrest and a second line
    representing cases that do not end in arrest.

    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''
    arrests_over_time = create_groups(df, 'arrest', 'date')
    plot = arrests_over_time.sort_index().cumsum().plot()
    plot.set_title('Running Total of Reported Cases by Arrest Status, 2017-2018')
    figure = plot.get_figure()
    figure.savefig('test.png', dpi=400)


def create_crime_location_heatmap(df, output_file):
    '''
    Creates a heatmap displaying number of reported cases by neighborhood
    (rows) as well as by type (columns).

    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''
    type_by_neighborhood = create_groups(df, 'primary_type', 'neighborhood')
    type_by_neighborhood.sort_values('THEFT', ascending=False, inplace=True)
    heatmap = sns.heatmap(type_by_neighborhood,
                          cmap='coolwarm',
                          fmt='g',
                          cbar=False,
                          linewidths=0.5,
                          center=2000,
                          mask=(type_by_neighborhood == 0))
    heatmap.set_title('Number of Crimes by Type and Neighborhood, 2017-2018')
    figure = heatmap.get_figure()
    figure.savefig('test.png', dpi=400)


def calculate_avg_arrests_per_month(df, output_file):
    '''
    Creates a summary table of the average number of arrests by neighborhood,
    per month, from 2017-2018. The table is sorted by avg. number of arrests.
    
    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''
    groups = df.groupby(['neighborhood'], as_index=False).agg({'arrest': 'sum'})
    groups.arrest = groups.arrest / 24
    groups = groups.round(2).sort_values('arrest', ascending=False)
    groups.index.name = None
    groups.rename(columns={'arrest': 'avg_arrests'}, inplace=True)
    groups.to_excel(output_file, index=False)


def prepare_acs_df(url=ACS):
    '''
    Use a census API to retrieve data from the American Community Survey on
    total population, percentage of black residents, percentage of residents
    with a high school diploma, median annual earnings, and percentage of
    residents below the poverty line, by zipcode.

    B01003_001E = total population
    B02001_003E = number of black residents
    B15003_017E = number of people with a high school diploma
    B08121_001E = median earnings in the past 12 months
    B07012_002E = number of people (one year ago) below poverty line

    Input:
        url (str): url prepared for the census API to pull 5 variables

    Output:
        acs_df (dataframe): dataframe containing acs details by zipcode
    '''
    request_obj = requests.get(url)
    json = request_obj.json()
    df = pd.DataFrame(json)
    header = df.iloc[0]
    df.rename(columns=header, inplace=True)
    df.drop([0], inplace=True)
    df.rename(columns={'B01003_001E': 'total_population',
                       'B02001_003E': 'num_black',
                       'B15003_017E': 'num_over_25_with_hs_diploma',
                       'B08121_001E': 'median_annual_earnings',
                       'B07012_002E': 'num_below_poverty_line',
                       'zip code tabulation area': 'zip_code'}, inplace=True)
    df = df.apply(pd.to_numeric)
    df['percent_black'] = df.num_black / df.total_population
    df['percent_hs_diploma'] = df.num_over_25_with_hs_diploma /\
                                        df.total_population
    df['percent_below_poverty_line'] = df.num_below_poverty_line /\
                                       df.total_population
    df.drop(columns=['num_black', 'num_over_25_with_hs_diploma',\
                     'num_below_poverty_line'], inplace=True)
    acs_df = df[['zip_code', 'total_population', 'median_annual_earnings',\
                 'percent_black', 'percent_hs_diploma',\
                 'percent_below_poverty_line']]

    return acs_df


def create_crime_zip_codes(crime_df):
    '''
    DOC STRING - convert blocks to zip codes to join in census data
    '''
    truncated = df.drop_duplicates(subset=['latitude', 'longitude', 'block']) 
    search = SearchEngine(simple_zipcode=True)
    crime_df['zip_code'] = crime_df.apply(lambda x:\
                                          search.by_coordinates(x.Latitude,
                                          x.Longitude)[0].zipcode, axis=1)
    return crime_df

def join_acs_to_crime(crime_df, acs_df):
    '''
    DOC STRING
    '''
    full_df = pd.merge(crime_df, acs_df, on='zip_code', how='left')

def go():
    '''
    DOC STRING
    '''

    crime_2017 = load_one_year(URL_2017)
    crime_2018 = load_one_year(URL_2018)
    df = crime_2017.append(crime_2018)
    df['date'] = df.date.astype('datetime64[M]')
    df['month'] = df.date.dt.to_period('M')

    names = pd.read_csv('community_area_names.csv')
    df['community_area'] = pd.to_numeric(df['community_area'], errors='coerce')
    names['community_area'] = names['community_area'].astype(float)
    df = pd.merge(df, names, on='community_area')

    sns.set(font_scale=0.5)
    summarize_crimes_by_type(df, 'crimes_by_type.xlsx')
    create_crime_heatmap(df, 'crime_heatmap.png')
    summarize_crimes_by_month(df, 'crime_total_by_month.xlsx')
    summarize_crimes_by_neighborhood(df, 'crime_and_arrest_total_by_neighborhood.xlsx')
    summarize_arrests_over_time(df, 'running_total_cases.png')
    create_crime_location_heatmap(df, 'crimes_by_neighborhood.png')
    calculate_avg_arrests_per_month(df, 'avg_arrests_per_month.xlsx')

    acs_df = prepare_acs_df()
    crime_zips_small = create_crime_zip_codes(df)



































