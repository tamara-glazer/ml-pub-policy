'''
Machine Learning Pipeline
1. Read/Load Data (CSV)
2. Explore Data
3. Pre-Process and Clean Data
4. Generate Features/Predictors
5. Build Machine Learning Classifiers
6. Evaluate Classifiers

Author: Tammy Glazer
Citation: https://github.com/rayidghani/magicloops/blob/master/simpleloop.py
'''
import warnings
import re
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_recall_curve


# Raw data CSV filename
RAW_DATA = 'projects_2012_2013.csv'
# Name of column to use as Index (specify '' if N/A)
UNIQUE_ID = 'projectid'
# Specify a single variable to view its distribution as a histogram
DISTRIBUTION = 'school_state'
# List two continuous variables to view a scatterplot (examine correlation)
CONTINUOUS_TWO = ['total_price_including_optional_support', 'students_reached']
# State true if outcome variable is binary, otherwise state False
BINARY = True
# State the name of the outcome variable (specify '' if N/A)
TARGET = 'funded_over_60'
# List any two categorical variables to view a heatmap (examine correlation)
CATEGORICAL_TWO = ['school_magnet', 'teacher_prefix']
# Specify a continuous and categorical variable to view a barplot
CATEGORICAL_VAR = 'primary_focus_subject'
CONTINUOUS_VAR = 'students_reached'
# Specify a NUMERIC attribute to print a list of outliers for that attribute
OUTLIER = 'students_reached'
# Specify a continuous feature to discretize
FEATURE = 'students_reached'
# Specify bins as inclusive integer tuples to discretize the above variable.
# Alternatively, BINS can be set to an integer. Comment out option not is use.
#BINS = pd.IntervalIndex.from_tuples([(20, 39), (40, 69), (70, 110)], closed='both')
BINS = 3
# Specify labels for discrete bins or leave list empty for default label
LABEL = []
# Specify a feature to create dummies
DUMMIES = 'grade_level'
# Specify test size and random state if not using temporal holdouts methodology
# TEST_SIZE = 0.2
# RANDOM_STATE = 1
# Select the number of features to use in models (using SelectKBest)
NUM_FEATURES = 10
# Specify threshold for evaluation metrics (if probability >= X, then 1)
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
# Specify the field to use for temporal validations
DATE_FIELD = 'date_posted'


# Set filenames to print tables/graphs/plots
DATATYPES = 'datatypes.csv'
RANGE = 'range_of_values.csv'
NULLS = 'null_values.csv'
CORRELATION_TABLE = 'correlation_table.csv'
CORRELATION_IMAGE = 'correlation_heatmap.png'
HISTOGRAM = 'histogram.png'
SCATTERPLOT = 'correlation.png'
HEATMAP = 'heatmap.png'
BARPLOT = 'barplot.png'
TREE = 'tree.png'


# Specify models to run
ANALYSIS = ['LR', 'KNN', 'DT', 'SVM', 'RF', 'BOOST', 'BAG']


# Define parameters for classifiers
MODELS = {'LR': LogisticRegression(penalty='l1',
                                   C=1e5,
                                   class_weight='balanced'),
          'KNN': KNeighborsClassifier(n_neighbors=3,
                                      weights='uniform',
                                      p=2),
          'DT': DecisionTreeClassifier(criterion='gini',
                                       max_depth=3,
                                       max_features=6),
          'SVM': LinearSVC(tol=1e-5,
                           C=1,
                           random_state=1),
          'RF': RandomForestClassifier(n_estimators=4,
                                       max_depth=6,
                                       n_jobs=1),
          'BOOST': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                      n_estimators=10,
                                      random_state=1),
          'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=1),
                                   n_estimators=10,
                                   random_state=1)}

GRID = {'LR': {'penalty': ['l1', 'l2'],
               'C': [0.00001, 1],
               'class_weight': ['balanced']},
        'KNN': {'n_neighbors': [1, 5],
                'weights': ['uniform'],
                'p': [1, 2]},
        'DT': {'criterion': ['gini'],
               'max_depth': [1, 5],
               'max_features': [2, 10]},
        'SVM': {'tol': [1e-5],
                'C': [0.00001, 1],
                'random_state': [1]},
        'RF': {'n_estimators': [5],
               'max_depth': [1, 4],
               'n_jobs': [1]},
        'BOOST': {'n_estimators': [1, 10],
                  'random_state': [1]},
        'BAG': {'n_estimators': [1, 10],
                'random_state': [1]}}


def read_data(file=RAW_DATA, unique_id=UNIQUE_ID):
    '''
    Reads in a CSV using pandas and converts the CSV into a pandas dataframe.
    Sets the index to a unique identifier if an attribute(s) is specified.

    Input:
        file (str): filename for a CSV

    Output:
        df (dataframe): a pandas dataframe
    '''
    df = pd.read_csv(file)
    if unique_id:
        df.set_index(unique_id, inplace=True)

    return df


def describe_data(df, file_1=DATATYPES, file_2=RANGE, file_3=NULLS,
                  file_4=CORRELATION_TABLE, file_5=CORRELATION_IMAGE):
    '''
    Describes the dataset as a whole by performaing the following actions:

    Prints:
        Whether the index is unique (if an index is specified)
    Writes CSVs:
        1. Describing the datatype of each variable
        2. Describing the range of values for each column
        3. Describing the number of null observations and % of total
           observations that are null for each variable
        4. Describing the correlation between each pair of variables
    Saves a heatmap:
        Displaying these correlations between variables

    Input:
        df (dataframe): a pandas dataframe
        file_1 (str): filename for the datatypes CSV
        file_2 (str): filename for the range/distribution CSV
        file_3 (str): filename for the nulls CSV
        file_4 (str): filename for the correlaton CSV
        file_5 (str): filename for the correlation heatmap PNG
    '''
    if UNIQUE_ID:
        if df.index.is_unique:
            print('The index represents unique entities')
        else:
            print('The index DOES NOT represent unique entities')

    df.dtypes.to_csv(file_1, header=False)
    df.describe().to_csv(file_2, header=True)
    nulls = df.isna().sum().to_frame('count_null')
    nulls['pct_null'] = (nulls['count_null'] / df.size).round(4)
    nulls.to_csv(file_3)

    df.corr().round(4).to_csv(file_4)

    plt.figure()
    sns.set(font_scale=0.5)
    heatmap = sns.heatmap(df.corr().round(2),
                          cmap='Blues',
                          annot=True,
                          annot_kws={'size': 5},
                          fmt='g',
                          cbar=False,
                          linewidths=0.5)
    heatmap.set_title('Correlation Between Variables')
    heatmap.figure.tight_layout()
    plt.savefig(file_5, dpi=400)
    plt.close()


def distribution(df, distrib=DISTRIBUTION, file_1=HISTOGRAM):
    '''
    Creates a histogram for any variable ignoring null values.
    Note that skewed distributions highlight the presence of outliers.

    Inputs:
        df (dataframe): a pandas dataframe
        distribution (str): a variable to visualize
        file_1 (str): filename to save the histogram
    '''
    plt.figure()
    sns.set(font_scale=0.75)
    sns.distplot(df[distrib].dropna(),
                 hist=True,
                 kde=False,
                 color='blue',
                 hist_kws={'edgecolor':'black'})
    plt.title('{} distribution'.format(distrib))
    plt.xlabel(distrib)
    plt.ylabel('Number of Observations')
    plt.savefig(file_1, dpi=400)
    plt.close()


def create_scatterplot(df, comparison=CONTINUOUS_TWO, file_1=SCATTERPLOT):
    '''
    Visualize the correlation between any two continuous variables on a
    scatterplot. Note: the target variable will appear on color only in the
    case of a binary target (eg. 0 or 1). Note that skewed plots highlight
    the presence of outliers.

    Inputs:
        df (dataframe): a pandas dataframe
        comparison (list): a list containing two continuous variables
        file_1 (str): filename to save the scatterplot
    '''
    x = comparison[0]
    y = comparison[1]

    plt.figure()
    sns.set(font_scale=0.75)
    if BINARY:
        correlation = sns.scatterplot(df[x], df[y], hue=TARGET, data=df)
    else:
        correlation = sns.scatterplot(df[x], df[y], data=df)
    correlation.set_title('Relationship between {} and {}'.format(x, y))
    plt.savefig(file_1, dpi=400)
    plt.close()


def create_heatmap(df, comparison=CATEGORICAL_TWO, file_1=HEATMAP):
    '''
    Visualize the correlation between any two categorical variables in a
    heatmap. This can be used to identify outliers as well.

    Inputs:
        df (dataframe): a pandas dataframe
        comparison (list): a list containing two categorical variables
        file_1 (str): filename to save the heatmap
    '''
    x = comparison[0]
    y = comparison[1]

    plt.figure()
    sns.set(font_scale=0.75)
    groups = df.pivot_table(index=x,
                            columns=y,
                            aggfunc='size')
    heatmap = sns.heatmap(groups,
                          cmap='Blues',
                          annot=True,
                          annot_kws={'size': 5},
                          fmt='g',
                          cbar=False,
                          linewidths=0.5,
                          mask=(groups == 0))
    heatmap.set_title('Number of observations by {} and {}'.format(x, y))
    plt.savefig(file_1, dpi=400)
    plt.close()


def create_barplot(df, cat=CATEGORICAL_VAR, con=CONTINUOUS_VAR,
                   file_1=BARPLOT):
    '''
    Visualize the relationship between a categorical variable and the average
    of a continuous variable through a barplot where x=categorical and
    y=continuous.

    Inputs:
        df (dataframe): a pandas dataframe
        cat (str): a categorical variable name
        con (str): a continuous variable name
        file_1 (str): filename to save the barplot
    '''
    x = cat
    y = con

    plt.figure()
    bars = sns.barplot(x, y, data=df, color='salmon', saturation=0.7, ci=None)
    bars.set_title('Average {} By {}'.format(y, x))
    plt.savefig(file_1, dpi=400)
    plt.close()


def find_outliers(df, outlier=OUTLIER):
    '''
    Identifies and prints a dataframe containing potential outliers defined
    as points that fall more than 1.5 times the interquartile range above
    the third quartile or below the first quartile for a given numeric column.

    Inputs:
        df (dataframe): a pandas dataframe
        column (str): a numeric attribute column

    Output [Print Statement]:
        df (dataframe): a list of potential outliers to examine
    '''
    col = df[outlier]
    quartile_1 = np.percentile(col, 25)
    quartile_3 = np.percentile(col, 75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (1.5 * iqr)
    upper_bound = quartile_3 + (1.5 * iqr)
    out = df.loc[(col > upper_bound) | (col < lower_bound)][outlier].to_frame()
    print(out)


def pre_process(df):
    '''
    For each float/numeric column, fills in missing values with the mean value
    for the column. For all other columns, drops rows with missing values.

    Input:
        df (dataframe): a pandas dataframe

    Output:
        df (dataframe): an updated pandas dataframe
    '''
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col].fillna(df[col].mean(), inplace=True)
    df.dropna(inplace=True)

    return df


def discretize_continuous_variable(df, feature=FEATURE, bins=BINS, label=LABEL):
    '''
    Creates a new column in the dataframe containing a categorical variable
    to represent a specified continuous variable. Bins can be set automatically
    or manually. Drops the original variable from the dataset.

    Inputs:
        df (dataframe): a pandas dataframe
        feature (str): the name of a continuous variable
        bins (int or pandas function): the bin size (set automatically or
                                                     manually)
        label (list): a list of labels (optional)

    Return:
        df (dataframe): dataframe with a new, discretized column
    '''
    if label:
        df[feature + '_bins'] = pd.cut(df[feature], bins=bins, labels=label)
    else:
        df[feature + '_bins'] = pd.cut(df[feature], bins=bins)
    df.drop([feature], axis=1, inplace=True)

    return df


def create_dummies(df, feature=DUMMIES):
    '''
    Takes a categorical variable and creates dummy variables from it,
    which are concatenated to the end of the dataframe. Drops the original
    variable from the dataset.

    Inputs:
        df (dataframe): a pandas dataframe
        feature (str): a variable name

    Output:
        df (dataframe): dataframe with new dummy variable columns
    '''
    dummies = pd.get_dummies(df[feature], prefix=feature)
    df = pd.concat([df, dummies], axis=1)
    df.drop([feature], axis=1, inplace=True)

    return df


def select_features(x_train, y_train, number_features=NUM_FEATURES):
    '''
    Computes the ch2 statistic between each feature of in the x-training set
    and the y-training set to determine which features provide the most
    information about y, and retains the K most explanatory features
    as specfied by number_features.

    Inputs:
        x_train (dataframe): a dataframe containing all feature options
        y_train (dataframe): the training target variable
        number_features (int): number of features to retain

    Output:
        features (series): a series containing the K best features
    '''
    best = SelectKBest(score_func=chi2, k=6)
    fit = best.fit(x_train, y_train)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(x_train.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['feature', 'score']
    largest = feature_scores.nlargest(number_features, 'score')

    return largest.feature


def generate_baseline(y_train, y_test, target=TARGET):
    '''
    Given a training set of features and outcome variable, determines the
    most frequent outcome label and applies this as the predicted value
    for every observation in the testing dataset. If this value is 1,
    assigns probability of predicting 1 to 1, and if this value is 0,
    assigns probabiliyt of predicting 1 to 0. This is used as a baseline and
    leverages the Zero Rule Algorithm.

    Inputs:
        y_train (dataframe): the outcome dataframe used for training
        y_test (dataframe): the outcome dataframe used for testing
        target (str): outcome variable name

    Outputs:
        y_predict (dataframe): the baseline outcome prediction
        y_predict_probability (int): probability that the outcome is 1
    '''
    val = int(y_train[target].mode())
    y_predict = y_test.copy()
    y_predict[target] = val
    y_predict_probability = y_test.copy()
    if val == 1:
        y_predict_probability = 1
    else:
        y_predict_probability[target] = 0

    return y_predict, y_predict_probability


def temporal_validation(df, start_train, end_train, start_test, end_test,
                        date_field=DATE_FIELD, target=TARGET):
    '''
    Creates testing and training dataframes using a temporal holdouts
    methodology. Given a specified start and end date for the training
    and testing datasets, divides the original dataframe accordingly.
    A rolling window (ie. 6 months) can be established by passing this
    function through a for-loop.

    Inputs:
        df (dataframe): clean dataframe
        start_train (date): start date for training data
        end_train (date): end date for training data
        start_test (date): start date for testing data
        end_test (date): end date for testing data
        date_field (str): name of the date feature on which to split
        target (str): outcome variable name

    Outputs:
        x_train (dataframe): training feature dataset
        x_test (dataframe): testing feature dataset
        y_train (dataframe): training outcome dataset
        y_test (dataframe): testing outcome dataset
    '''
    train = df[(df[date_field] >= start_train) & (df[date_field] <= end_train)]
    test = df[(df[date_field] >= start_test) & (df[date_field] <= end_test)]

    features = df.iloc[0].index.to_list()
    features.remove(target)
    features.remove(date_field)
    x_train = train[features]
    y_train = train[[target]]
    x_test = test[features]
    y_test = test[[target]]

    return x_train, x_test, y_train, y_test


def run_models(x_train, x_test, y_train, y_test, dates, results,
               models_to_run=ANALYSIS, models=MODELS, grid=GRID,
               thresholds=THRESHOLDS):
    '''
    Given training and testing data, an empty dataframe, specified models,
    specified parameters, and a set of evaluation thresholds, conducts a
    complete machine learning analysis and outputs all evaluation metrics
    to a CSV. A parameter can be set to run one or more classifiers, and
    metrics can be calculated at different levels given any number of
    thresholds.

    Inputs:
        x_train (dataframe): training feature dataframe
        x_test (dataframe): testing feature dataframe
        y_train (dataframe): training outcome dataframe
        y_test (dataframe): testing outcome dataframe
        dates (str): time-span of the testing dataframe (for output table)
        results (dataframe): the dataframe that will be written to CSV
        models_to_run (lst): list of models to run
        models (dict): dictionary initializing all relevant models
        grid (dict): dictionary specifying all model parameters to use
        thresholds (lst): list of thresholds to calculate precision and recall

    Output:
        results (dataframe): the final table summarizing results and metrics
    '''
    # Prepare table
    lst = []
    for threshold in thresholds:
        for score in ['precision_{}', 'recall_{}']:
            label = score.format(threshold)
            lst.append(label)
    columns = ('test_dates', 'model_type', 'model', 'train_data_size',
               'test_data_size', 'parameters', 'AUC', 'accuracy_score',
               'f1_score') + tuple(lst)
    if results.empty:
        results = pd.DataFrame(columns=columns)
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    limited_features = select_features(x_train, y_train)
    x_train = x_train[limited_features]
    x_test = x_test[limited_features]
    train_data_size = len(x_train)
    test_data_size = len(x_test)


    # Generate Baseline
    y_predict, y_predict_probability = generate_baseline(y_train, y_test)
    row_base = [dates, 'baseline', 'most_frequent_train_label',\
                train_data_size, test_data_size, 'N/A',\
                roc_auc_score(y_test, y_predict_probability),\
                accuracy_score(y_test, y_predict),\
                f1_score(y_test, y_predict)]
    for threshold in thresholds:
        row_base.append(precision_score(y_test, y_predict))
        row_base.append(recall_score(y_test, y_predict))
    results.loc[len(results)] = row_base


    # Generate models
    for index, model in enumerate([models[x] for x in models_to_run]):
        parameters = grid[models_to_run[index]]
        for p in ParameterGrid(parameters):
            try:
                model.set_params(**p)
                trained_model = model.fit(x_train, y_train.values.ravel())
                if models_to_run[index] == 'SVM':
                    y_pred_probs = trained_model.decision_function(x_test)
                else:
                    y_pred_probs = trained_model.predict_proba(x_test)[:, 1]

                general_predict = (y_pred_probs >= 0.5).astype('int')

                row = [dates, models_to_run[index], model,\
                       train_data_size, test_data_size, p,\
                       roc_auc_score(y_test, general_predict),\
                       accuracy_score(y_test, general_predict),\
                       f1_score(y_test, general_predict)]

                for threshold in thresholds:
                    y_predicted_t = (y_pred_probs >= threshold).astype('int')
                    row.append(precision_score(y_test, y_predicted_t))
                    row.append(recall_score(y_test, y_predicted_t))

                results.loc[len(results)] = row

            except IndexError as e:
                print('Error', e)
                continue

    return results
