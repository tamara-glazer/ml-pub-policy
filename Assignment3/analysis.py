'''
Uses a machine learning pipeline to train a variety of machine learning
models to predict if projects on Donorschoose will NOT get fully
funded within 60 days of posting. Calculates several evaluation metrics
to better understand the strength of each model and validate assumptions.

Note: Global variables in the ml_pipeline.py file are currently set to the
Donorschoose dataset. These values can easily be modified to conduct an
analysis on a new dataset.

Author: Tammy Glazer
'''
import pandas as pd
import ml_pipeline as m


def explore_and_process_data():
    '''
    Reads/loads the Donorschoose dataset, conducts data exploration, and
    pre-processes the data. Note that the raw data file is specified as a
    global variable in the ml_pipeline.py file. Creates an outcome variable
    called "funded_over_60" representing whether a project was not fully
    funded within 60 days of posting (1) or if it was funded in this
    time frame (0). Drops variables that are not necessary for the analysis,
    such as specific teacher account IDs.

    Output:
        df (dataframe): a processed pandas dataframe ready for analysis
    '''
    df = m.read_data()
    df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
    df['date_posted'] = pd.to_datetime(df['date_posted'])
    df.drop(['teacher_acctid', 'schoolid', 'school_ncesid', 'school_latitude',\
             'school_longitude', 'secondary_focus_area',\
             'secondary_focus_subject', 'school_city', 'school_district',
             'primary_focus_area', 'school_county', 'school_state'],
            axis=1, inplace=True)
    categorical_vars = ['school_metro', 'school_charter', 'school_magnet',
                        'teacher_prefix', 'primary_focus_subject',\
                        'resource_type', 'poverty_level', 'grade_level',
                        'eligible_double_your_impact_match']
    df[categorical_vars] = df[categorical_vars].astype('category')
    df['funded_over_60'] = (df['datefullyfunded'] -
                            df['date_posted']).dt.days > 60
    df['funded_over_60'] = df['funded_over_60'] * 1

    # Export descriptive tables and visualizations; initialize as needed
    # m.describe_data(df)

    df = m.pre_process(df)

    return df


def analyze_data(df, file_name='models.csv'):
    '''
    Using the pre-processed dataframe, keeps relevant features, discretizes
    continuous variables based on specified bins, converts categorical
    variables into dummies, drops irrelevant columns, and trains specified
    machine learning models to predict an outcome variable. Training
    and testing datasets are created using a rolling window of 6 months,
    providing 3 test sets. Exports baseline predictions along with all models,
    parameters, and evaluation metrics to a CSV. Specifically, trains
    models to predict whether a project on Donorschoose will not get funded
    within 60 days of posting and calculates evaluation metrics at the
    following thresholds: 1%, 2%, 5%, 10%, 20%, 30%, and 50%.

    Inputs:
        df (dataframe): pre-processed pandas dataframe
        filename (str): output filename
    '''
    continuous = ['total_price_including_optional_support', 'students_reached']
    discrete = ['school_metro', 'school_charter',\
                'school_magnet', 'teacher_prefix', 'primary_focus_subject',\
                'resource_type', 'poverty_level', 'grade_level',
                'eligible_double_your_impact_match',\
                'total_price_including_optional_support_bins',\
                'students_reached_bins']
    df.drop(['datefullyfunded'], axis=1, inplace=True)
    pd.options.mode.chained_assignment = None
    for var in continuous:
        df = m.discretize_continuous_variable(df, feature=var)
    for var in discrete:
        df = m.create_dummies(df, feature=var)

    time_range =\
    [['2012-01-01 00:00:00', '2012-06-30 00:00:00', '2012-07-01 00:00:00',\
     '2012-12-31 00:00:00'], ['2012-01-01 00:00:00', '2012-12-31 00:00:00',\
     '2013-01-01 00:00:00', '2013-06-30 00:00:00'], ['2012-01-01 00:00:00',\
     '2013-06-30 00:00:00', '2013-07-01 00:00:00', '2013-12-31 00:00:00']]

    results = pd.DataFrame()
    for dates in time_range:
        test_dates = '{} to {}'.format(dates[2], dates[3])
        x_train, x_test, y_train, y_test = m.temporal_validation(df,
                                                                 dates[0],
                                                                 dates[1],
                                                                 dates[2],
                                                                 dates[3])
        results = m.run_models(x_train, x_test, y_train, y_test, test_dates,
                               results=results)

    results.to_csv(file_name, index=False)


def run_pipeline():
    '''
    Runs the full machine learning pipeline

    Output:
        file (CSV): exports a table summarizing results and evaluation metrics
    '''
    df = explore_and_process_data()
    analyze_data(df)


if __name__ == "__main__":
    run_pipeline()
