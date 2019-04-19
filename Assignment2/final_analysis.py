'''
Uses a machine learning pipeline to train a Decision Tree model to predict who
will experience financial distress in the next two years with a degree
of accuracy

Note: Global variables in the ml_pipeline.py file are currently set to the
Credit dataset. These values can easily be modified to conduct an analysis
on a new dataset.

Author: Tammy Glazer
'''
import pandas as pd
import ml_pipeline as m


def explore_and_process_data(new_file=None):
    '''
    Reads/loads the Credit Dataset, conducts data exploration (including
    exporting several figures), and pre-processes the data. Note that the
    raw data file is specified as a global variable in the ml_pipeline.py
    file. Alternatively, a new filename can be passed into this function
    to override the default specification.

    Input:
        new_file (str): optional filename (otherwise specify in ml_pipeline.py)

    Output:
        df (dataframe): a processed pandas dataframe ready for analysis
    '''
    if not new_file:
        df = m.read_data()
    else:
        df = m.read_data(file=new_file)
    m.describe_data(df)
    for var in df.columns:
        m.distribution(df, distrib=var, file_1='{}_histogram.png'.format(var))
    m.create_scatterplot(df, comparison=['age', 'DebtRatio'],
                         file_1='age_debtratio.png')
    m.create_scatterplot(df, comparison=['age', 'MonthlyIncome'],
                         file_1='age_income.png')
    m.create_heatmap(df, comparison=['NumberOfDependents', \
                                     'NumberRealEstateLoansOrLines'],
                     file_1='dependents_loans.png')
    m.create_heatmap(df, comparison=['NumberOfDependents', 'zipcode'],
                     file_1='dependents_zipcodes.png')
    m.create_barplot(df, cat='NumberOfDependents', con='MonthlyIncome',
                     file_1='dependents_income.png')
    for var in df.columns:
        m.find_outliers(df, outlier=var)
    df = m.pre_process(df)

    return df


def analyze_data(df):
    '''
    Using the processed dataframe, keeps relevant features, discretizes
    continuous variables based on specified bins, converts categorical
    variables into dummies, drops irrelevant columns, and trains a decision
    tree to predict a specified outcome variable using a training dataset.
    Accuracy score can be obtained through the accuracy property of the
    Classifier object.

    Input:
        df (dataframe): pre-processed pandas dataframe

    Output:
        tree (obj): Classifier object (trained decision tree)
    '''
    truncated = df[['age', 'zipcode', 'SeriousDlqin2yrs']]
    pd.options.mode.chained_assignment = None
    truncated = m.discretize_continuous_variable(truncated)
    truncated = m.create_dummies(truncated, feature='age_bins')
    truncated = m.create_dummies(truncated, feature='zipcode')
    truncated.drop(columns=['age', 'age_bins', 'zipcode'], inplace=True)
    tree = m.buildtree(truncated)

    return tree


def run_pipeline(new_file=None):
    '''
    Runs the full machine learning pipeline

    Input:
        new_file (optional): raw data filename (alternatively, specify in
                             ml_pipeline.py)
    Output:
        tree (obj): trained decision tree
    '''
    if not new_file:
        df = explore_and_process_data()
    else:
        df = explore_and_process_data(new_file)
    tree = analyze_data(df)

    return tree


if __name__ == "__main__":
    run_pipeline()
