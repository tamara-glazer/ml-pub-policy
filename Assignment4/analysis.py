'''
Uses a machine learning pipeline to train a variety of machine learning
models to predict if projects on Donorschoose will NOT get fully
funded within 60 days of posting. Calculates several evaluation metrics
to better understand the strength of each model and validate assumptions.

Note: Global variables in the config.py file are currently set to the
Donorschoose dataset. These values can easily be modified to conduct an
analysis on a new dataset.

Author: Tammy Glazer
'''
import pandas as pd
from datetime import timedelta
from dateutil import relativedelta
from sklearn.preprocessing import MinMaxScaler
import config as c
import ml_pipeline as m


def prepare_full_dataset():
    '''
    Reads/loads the Donorschoose dataset and prepares the outcome variable
    field. Note that the raw data file is specified as a variable in config.py.
    The outcome variable, "funded_over_60", represents whether a project was
    not fully funded within 60 days of posting (1) or if it was funded in this
    time frame (0).

    Output:
        df (dataframe): an updated full pandas dataframe
    '''
    df = m.read_data()
    df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
    df['date_posted'] = pd.to_datetime(df['date_posted'])
    df['funded_over_60'] = (df['datefullyfunded'] -
                            df['date_posted']).dt.days > 60
    df['funded_over_60'] = df['funded_over_60'] * 1

    return df


def process_donor_data(x_train, x_test):
    '''
    Drop columns that will not be used as features (eg. IDs), change datatypes
    for training and testing data, process x_train, and finally process
    x_test SEPARATELY, ensuring that new fields that appear in x_test are
    categorized as 'Other'. Use value ranges in x_train to scale x_test,
    using MinMaxScaler(). The following line of code can be inserted into
    this function to produce descriptive tables and visualizations:
    m.describe_data(x_data)

    Inputs:
        x_train (dataframe): training features
        x_test (dataframe): testing features

    Output:
        x_train (dataframe): processed training features
        x_test (dataframe): processed testing features
    '''
    # Perform data type transformations applicable to both x_train and x_test
    for x_data in [x_train, x_test]:
        x_data.drop(c.DROP, axis=1, inplace=True)
        x_data[c.CATEGORICAL_VARS_FEW] = \
                x_data[c.CATEGORICAL_VARS_FEW].astype('category')
        x_data[c.CATEGORICAL_VARS_MANY] = \
                x_data[c.CATEGORICAL_VARS_MANY].astype('category')
        x_data[c.CONTINUOUS_VARS] = x_data[c.CONTINUOUS_VARS].astype(float)

    # Process x_train and x_test separately; scale continuous variables
    x_train = m.impute(x_train)
    x_test = m.impute(x_test)
    for var in c.CATEGORICAL_VARS_FEW:
        x_train, top_20 = m.create_dummies(x_train, feature=var)
        x_test = m.create_test_dummies(x_test, top_20, feature=var)
    for var in c.CATEGORICAL_VARS_MANY:
        x_train, top_20 = m.create_limited_dummies(x_train, feature=var)
        x_test = m.create_test_dummies(x_test, top_20, feature=var)
    min_max_scaler = MinMaxScaler()
    x_train[c.CONTINUOUS_VARS] = min_max_scaler.fit_transform(x_train[c.CONTINUOUS_VARS])
    x_test[c.CONTINUOUS_VARS] = min_max_scaler.transform(x_test[c.CONTINUOUS_VARS])

    return x_train, x_test


def analyze_data(df, file_name='models.csv', plot=False, grid=c.SMALL_GRID):
    '''
    Given a dataframe, keeps relevant features, converts categorical
    variables into dummies separately for testing and training data,
    and trains machine learning models to predict an outcome variable.
    Training and testing datasets are created using a rolling window of 6
    months, providing 3 test sets, with 60 days between training and testing
    sets. Exports baseline predictions along with all models, parameters, and
    evaluation metrics to a CSV. Specifically, trains models to predict
    whether a project on Donorschoose will not get funded within 60 days of
    posting and calculates evaluation metrics at for several population
    percentages (k-values).

    Note: To save precision-recall plots, update parameter 'plot' to True.

    Inputs:
        df (dataframe): pre-processed pandas dataframe
        filename (str): output filename
        plot (boolean): whether to save precision-recall plots for each model
    '''
    results = pd.DataFrame()

    # Temporal holdout train/test split (60-day gap b/w training & testing)
    end_date = c.END_DATE
    count = 2
    for i in range(c.HOLDOUTS):
        x_train, x_test, y_train, y_test,\
        start_test, end_test = m.temporal_validation(df, end_date=end_date)

        # pre-process x_train and x_test separately
        x_train, x_test = process_donor_data(x_train, x_test)
        missing_cols = set(x_train.columns) - set(x_test.columns)
        for col in missing_cols:
            x_test[col] = 0
        x_test = x_test[x_train.columns]

        # run models
        test_dates = '{} to {}'.format(start_test, end_test)
        results, past_count = m.run_models(x_train, x_test, y_train, y_test,
                                           test_dates, results=results,
                                           grid=grid, create_plot=plot,
                                           counter=count)
        end_date = end_date - relativedelta.relativedelta(months=c.ROLLING_MONTHS)
        count = past_count

    results.to_csv(file_name, index=False)


def run_pipeline(plot=False, grid=c.SMALL_GRID):
    '''
    Runs the full machine learning pipeline. Pass parameter 'plot=True' to
    locally save precision-recall plots, default=False. Pass parameter
    'grid=c.LARGE_GRID' to run the large grid (default=c.SMALL_GRID)

    Inputs:
        plot (boolean): whether to locally save precision-recall plots
        grid (dict): dictionary specifying all model parameters to use

    Output:
        file (CSV): exports a table summarizing results and evaluation metrics
    '''
    df = prepare_full_dataset()
    analyze_data(df, plot=plot, grid=grid)


if __name__ == "__main__":
    run_pipeline()
