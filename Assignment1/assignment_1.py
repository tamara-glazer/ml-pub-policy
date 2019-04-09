'''
DESCRIPTION

 ##remove slice warning##

Author: Tammy Glazer
'''

import numpy as np
import pandas as pd
import re
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
    crimes_by_type.sort_values('total', ascending=False, inplace=True)
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
    figure.savefig(output_file, dpi=400)
    plt.clf()


def summarize_crimes_by_month(df, output_file):
    '''
    Creates a summary table of total number of reported cases and average
    number of arrests per month, from 2017-2018.

    Inputs:
        df (dataframe): a pandas dataframe
        output_file (str): output file name
    '''
    monthly = df.groupby(['month'], as_index=False).agg({'case_number':
                                                         'count', 'arrest':
                                                         'sum'})
    monthly.index.name = None
    monthly.rename(columns={'case_number': 'number_of_cases', 'arrest':
                            'number_of_arrests'}, inplace=True)
    monthly.to_excel(output_file, index=False)


def summarize_crimes_by_neighborhood(df, output_file_high, output_file_low):
    '''
    Creates a summary table of total number of cases and arrests reported by
    neighborhood in 2017-2018. Two tables are saved: one for the top 5 most
    reported crimes, and one for the top 5 fewest reported crimes.
    
    Inputs:
        df (dataframe): a pandas dataframe
        output_file_high (str): output file name (5 highest)
        output_file_low (str): output file name (5 lowest)
    '''

    district = df.groupby(['neighborhood'], as_index=False).agg({'case_number':
                                                                 'count',
                                                                 'arrest':
                                                                 'sum'})
    district.index.name = None
    district.rename(columns={'case_number': 'number_of_cases', 'arrest':
                             'number_of_arrests'}, inplace=True)
    highest = district.nlargest(5, 'number_of_cases')
    lowest = district.nsmallest(5, 'number_of_cases')
    highest.to_excel(output_file_high, index=False)
    lowest.to_excel(output_file_low, index=False)
    plt.clf()


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
    figure.savefig(output_file, dpi=400)


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
    figure.savefig(output_file, dpi=400)
    plt.clf()


def print_summary_tables_q1(df):
    '''
    Print all summary tables and graphs for Question 1.

    Input:
        df (dataframe): crimes dataframe
    '''
    summarize_crimes_by_type(df, 'crimes_by_type.xlsx')
    create_crime_heatmap(df, 'crime_heatmap.png')
    summarize_crimes_by_month(df, 'crimes_by_month.xlsx')
    summarize_crimes_by_neighborhood(df, 'top5_neighborhoods.xlsx',
                                     'bottom5_neighborhoods.xlsx')
    summarize_arrests_over_time(df, 'running_total_cases.png')
    create_crime_location_heatmap(df, 'crimes_by_neighborhood.png')


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
    acs_df.loc[:,'zip_code'] = acs_df.astype({'zip_code': float})

    return acs_df


def create_crime_zip_codes(crime_df):
    '''
    Use the uszipcode library to identify a zip code for each unique
    pair of latitude and longitude coordinates in the Crime Dataset.
    Merges zip code information back into the Crime Dataset to later join with
    ACS data.

    Input:
        crime_df (dataframe): original crime dataframe

    Output:
        crime_df (dataframe): new crime dataframe including zip codes
    '''
    crime_df.loc[:,'latitude'] = truncated.latitude.astype(float)
    crime_df.loc[:,'longitude'] = truncated.longitude.astype(float)
    truncated = crime_df.drop_duplicates(subset=['latitude', 'longitude',
                                                 'block'])
    truncated = truncated[['block', 'latitude', 'longitude']]
    truncated = truncated.dropna()
    search = SearchEngine(simple_zipcode=True)
    truncated['zip_code'] = truncated.apply(lambda x:
                                            search.by_coordinates(x['latitude'],
                                            x['longitude'])[0].zipcode, axis=1)
    merged_df = pd.merge(crime_df, truncated, on=['block', 'latitude',\
                                                  'longitude'], how='left')
    merged_df.loc[:, 'zip_code'] = pd.to_numeric(merged_df['zip_code'],
                                                 errors='coerce')

    return merged_df


def calculate_chicago_stats(acs_df, output_file):
    '''
    Calculates descriptive statistics for 2017-2018 demographics in Chicago
    at the zip_code level, including mean, min, 25%, 50%, 75%, and maximum
    percent of people below the poverty line, median annual earnings, and
    percent of people who are black.

    Inputs:
        acs_df (dataframe): ACS data
        output_file (str): output filename
    '''
    chicago_zip_codes = pd.read_excel('chicago_zip_codes.xlsx')
    chicago_data = pd.merge(chicago_zip_codes, acs_df, on=['zip_code'],
                            how='left')
    chicago_data.dropna()
    summary = chicago_data.describe()[['percent_below_poverty_line',
                                       'median_annual_earnings',
                                       'percent_black']]
    stats = summary.loc[['mean', 'min', '25%', '50%', '75%', 'max'],:].round(2)
    stats.to_excel(output_file)


def specific_crime_reports(full_df, output_file, crime_type):
    '''
    Calculates descriptive statistics at the zip code level among zip
    codes where a specified type of crime took place, including mean, min,
    25%, 50%, 75%, and maximum percent of people below the poverty line,
    median annual earnings, and percent of people who are black.

    Inputs:
        full_df (dataframe): full Crime data joined with ACS data
        output_file (str): output filename
        crime_type (str): type of crime
    '''
    full_df_crime = full_df[full_df.primary_type == crime_type]
    summary = full_df_crime.describe()[['percent_below_poverty_line',
                                        'median_annual_earnings',
                                        'percent_black']]
    stats = summary.loc[['mean', 'min', '25%', '50%', '75%', 'max'],:].round(2)
    stats.to_excel(output_file)


def changes_in_crime_over_time(full_df, output_file, demographic, crime_type):
    '''
    Creates a line plot to demonstrate how average values for a specified
    demographic change over time (by month) in locations where a specified
    crime type was reported between 2017-2018. Line plots are constructed at
    the zip code level to demonstrate trends.

    Inputs:
        full_df (dataframe): full Crime data joined with ACS data
        output_file (str): output filename
        demographic (str): ACS demographic (y-axis)
        crimetype (str): type of crime to visualize
    '''
    full_df_crime = full_df[full_df.primary_type == crime_type]
    summary = full_df_crime[['date', demographic]]
    table = summary.groupby('date').mean()
    
    plt.plot(table.index, table[demographic])
    plt.suptitle('Demographics in Zip Codes where ' + crime_type + ' occurs: '
                 + demographic)
    plt.xlabel('date')
    plt.ylabel(demographic)
    plt.savefig(output_file, dpi=400)
    plt.clf()


def print_summary_tables_q2(acs_df, full_df):
    '''
    Print all summary tables for Question 2.

    Input:
        acs_df (dataframe): ACS data
        full_df (dataframe): Crime data joined with ACS data on zip code
    '''
    calculate_chicago_stats(acs_df, 'chicago_stats.xlsx')
    specific_crime_reports(full_df, 'battery_reports.xlsx', 'BATTERY')
    specific_crime_reports(full_df, 'homicide_reports.xlsx', 'HOMICIDE')
    changes_in_crime_over_time(full_df, 'battery_black_over_time.png',
                               'percent_black', 'BATTERY')
    changes_in_crime_over_time(full_df, 'battery_earnings_over_time.png',
                               'median_annual_earnings', 'BATTERY')
    changes_in_crime_over_time(full_df, 'battery_poverty_line_over_time.png',
                               'percent_below_poverty_line', 'BATTERY')
    changes_in_crime_over_time(full_df, 'homicide_black_over_time.png',
                               'percent_black', 'HOMICIDE')
    changes_in_crime_over_time(full_df, 'homicide_earnings_over_time.png',
                               'median_annual_earnings', 'HOMICIDE')
    changes_in_crime_over_time(full_df, 'homicide_poverty_line_over_time.png',
                               'percent_below_poverty_line', 'HOMICIDE')
    specific_crime_reports(full_df, 'deceptive_practice_reports.xlsx',
                           'DECEPTIVE PRACTICE')
    specific_crime_reports(full_df,
                           'sex_offense_reports.xlsx', 'SEX OFFENSE')


def calculate_q3_part_1(full_df):
    '''
    Compute metrics on how crime has changed in Chicago from 2017 to 2018.
    Note that these metrics are not exported to an output file.

    Input:
        full_df (dataframe): full dataframe
    '''
    change_over_time = full_df.groupby('year').agg({'case_number': 'count',
                                                    'arrest': 'sum'})
    change_over_time['percent_crime_arrest'] = (change_over_time.arrest /
                                               change_over_time.case_number)
    full_df_battery = full_df[full_df.primary_type == battery]
    summary_1 = full_df_battery[['year', 'percent_below_poverty_line']]
    battery_poverty_line = summary1.groupby('year').mean()   
    summary_2 = full_df_battery[['year', 'percent_black']]
    battery_percent_black = summary2.groupby('year').mean()
    summary_3 = full_df_battery[['year', 'median_annual_earnings']]
    battery_earnings = summary3.groupby('year').mean() 


def calculate_YOY_comparison(full_df, output_file):
    '''
    Compute how crime has changed between 2017 and 2017 in Chicago over the
    same 28 day period (June 28th to July 25th).

    Inputs:
        full_df (dataframe): full dataframe
        output_file (str): output filename
    '''
    full_df['date'] = pd.to_datetime(full_df['date'])
    truncated_2017 = full_df[(full_df['date'] >= '2017-6-28') & (full_df['date']
                                                                <= '2017-7-25')]
    truncated_2018 = full_df[(full_df['date'] >= '2018-6-28') & (full_df['date']
                                                                <= '2018-7-25')]
    df_limited = truncated_2017.append(truncated_2018)
    table = create_groups(df_limited, 'year', 'primary_type')
    table = table.append(pd.Series(table.sum(), name='TOTAL'))
    table['percent_change'] = table.pct_change(axis='columns')['2018'].round(2)
    table.to_excel(output_file)


def calculate_YTD_comparison(full_df, output_file):
    '''
    Compute how crime has changed between 2017 and 2017 in Chicago over the
    same year-to-date period (June 1st to July 25th).

    Inputs:
        full_df (dataframe): full dataframe
        output_file (str): output filename
    '''
    full_df['date'] = pd.to_datetime(full_df['date'])
    truncated_2017 = full_df[(full_df['date'] >= '2017-1-1') & (full_df['date']
                                                               <= '2017-7-25')]
    truncated_2018 = full_df[(full_df['date'] >= '2018-1-1') & (full_df['date']
                                                               <= '2018-7-25')]
    table = create_groups(df_limited, 'year', 'primary_type')
    table = table.append(pd.Series(table.sum(), name='TOTAL'))
    table['percent_change'] = table.pct_change(axis='columns')['2018'].round(2)
    table.to_excel(output_file)


def print_summary_tables_q3(df):
    '''
    Print all summary tables for Question 3.

    Input:
        df (dataframe): crimes dataframe
    '''
    calculate_YOY_comparison(df, 'YOY_comparison.xlsx')
    calculate_YTD_comparison(df, 'YTD_comparison.xlsx')


def calculate_crime_probability(df, output_file):
    '''
    Calculate the most likely crime type given a call comes from 2111
    S Michigan Avenue along with the probabilities of each type of request.

    Inputs:
        df (dataframe): crime dataframe
        output_file (str): output file
    '''
    df.block = df.block.astype(str)
    df["correct_blocks"] = df['block'].str.contains('S\sMICHIGAN\sAVE',
                                                    regex=True)
    df = df.drop(df[df.correct_blocks == False].index)
    likely = df.groupby(['primary_type'], as_index=False).agg({'case_number':
                        'count'}).sort_values('case_number', ascending=False)
    total = likely.case_number.sum()
    likely['probability'] = (likely['case_number'] / total).round(3)
    likely.rename(columns={'case_number': 'number_of_cases'}, inplace=True)
    likely.to_excel(output_file, index=False)


def calculate_call_likelihood(df, output_file):
    '''
    Calculates the likelihood that a call about Theft is coming from Garfield
    Park vs. Uptown

    Inputs:
        df (dataframe): crime dataframe
        output_file (str): output file
    '''
    groups = create_groups(df, 'primary_type', 'neighborhood')
    groups = groups[['THEFT']].sort_values('THEFT', ascending=False)
    total = groups.THEFT.sum()
    groups = groups.loc[['West Garfield Park', 'East Garfield Park', 'Uptown']]
    groups['probability'] = groups.THEFT / total
    groups.sort_values('probability', ascending=False, inplace=True)
    groups.to_excel(output_file, index=False)


def print_summary_tables_q4(df):
    '''
    Print all summary tables for Question 4.

    Input:
        df (dataframe): crimes dataframe
    '''
    calculate_crime_probability(df, 'crime_probability.xlsx')
    calculate_call_likelihood(df, 'calculate_call_likelihood.xlsx')


def complete_analysis():
    '''
    Conducts the complete analysis, including downloading reported crime data
    from the Chicago Open Data Portal for 2017 and 2018, generating summary
    statistics for the crime reports (summary_tables_q1), joining in data
    containing neighborhood names, generating zip codes based on lat/long
    information in the crime dataset and joining ACS data on zip code,
    and conducing analysis on this augmented dataset (summary_tables_q2,
    summary_tables_q3, summary_tables_q4).
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
    print_summary_tables_q1(df)

    acs_df = prepare_acs_df()
    crime_df = create_crime_zip_codes(df)
    full_df = pd.merge(crime_df, acs_df, on='zip_code', how='left')

    print_summary_tables_q2(acs_df, full_df)
    print_summary_tables_q3(df)
    print_summary_tables_q4(df)
