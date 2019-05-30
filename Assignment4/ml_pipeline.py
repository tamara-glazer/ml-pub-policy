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
from datetime import timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dateutil import relativedelta
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score,\
                            recall_score, f1_score, precision_recall_curve
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import config as c


def read_data(file=c.RAW_DATA, unique_id=c.UNIQUE_ID):
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


def describe_data(df, file_1=c.DATATYPES, file_2=c.RANGE, file_3=c.NULLS,
                  file_4=c.CORRELATION_TABLE, file_5=c.CORRELATION_IMAGE):
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
    if c.UNIQUE_ID:
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


def distribution(df, distrib=c.DISTRIBUTION, file_1=c.HISTOGRAM):
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


def create_scatterplot(df, comparison=c.CONTINUOUS_TWO, file_1=c.SCATTERPLOT):
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
    if c.BINARY:
        correlation = sns.scatterplot(df[x], df[y], hue=c.TARGET, data=df)
    else:
        correlation = sns.scatterplot(df[x], df[y], data=df)
    correlation.set_title('Relationship between {} and {}'.format(x, y))
    plt.savefig(file_1, dpi=400)
    plt.close()


def create_heatmap(df, comparison=c.CATEGORICAL_TWO, file_1=c.HEATMAP):
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


def create_barplot(df, cat=c.CATEGORICAL_VAR, con=c.CONTINUOUS_VAR,
                   file_1=c.BARPLOT):
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


def find_outliers(df, outlier=c.OUTLIER):
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


def impute(df):
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
        else:
            df[col] = df[col].cat.add_categories(['missing'])
            df[col].fillna('missing', inplace=True)

    return df


def discretize_continuous_variable(df, feature=c.FEATURE, bins=c.BINS,
                                   label=c.LABEL):
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


def create_dummies(df, feature=c.DUMMIES):
    '''
    Takes a categorical variable and creates dummy features from it,
    which are concatenated to the end of the dataframe. Drops the original
    variable from the dataset.

    Inputs:
        df (dataframe): a pandas dataframe
        feature (str): a variable name

    Output:
        df (dataframe): dataframe with new dummy variable columns
        top_20 (lst): a list of the dummy columns categories
    '''
    top_20 = df[feature].unique().tolist()
    dummies = pd.get_dummies(df[feature], prefix=feature)
    df = pd.concat([df, dummies], axis=1)
    df.drop([feature], axis=1, inplace=True)
    other_col = feature + '_' + 'Other'
    if other_col not in df.columns:
        df[other_col] = 0

    return df, top_20


def create_limited_dummies(df, feature=c.DUMMIES):
    '''
    Takes a categorical variable with more than 20 categories and creates up
    to 21 dummy features from it, which are concatenated to the end of the
    dataframe. Drops the original variable from the dataset. The 20 features
    selected are those with the most occurances in the training data. All
    other features are categorized as "Other".

    Inputs:
        df (dataframe): x_train dataframe
        feature (str): a variable name

    Output:
        updated_df (dataframe): dataframe with new dummy variable columns
        top_20 (lst): a list of the dummy columns categories
    '''
    top_20 = df[feature].value_counts()[:20].index.tolist()
    new_col = feature + '_new'
    df[new_col] = 'Other'
    df.loc[df[df[feature].isin(top_20)].index, new_col] = df[feature]
    df.drop([feature], axis=1, inplace=True)
    df.rename({new_col:feature}, axis=1, inplace=True)
    updated_df, top_20 = create_dummies(df, feature=feature)

    other_col = feature + '_' + 'Other'
    if other_col not in updated_df.columns:
        df[other_col] = 0

    return updated_df, top_20


def create_test_dummies(x_test, top_20, feature=c.DUMMIES):
    '''
    Creates dummy columns for x_test by including the same dummy columns
    as appear in x_train and categorizing records that do not fit into these
    categories as "Other". Drops the original variable from the dataset.

    Inputs:
        x_test (dataframe): x_test dataframe
        top_20 (lst): a list of the dummy column categories from x_train
        feature (str): a variable name

    Output:
        x_test (dataframe): updated x_test dataframe with dummy columns
    '''
    for val in top_20:
        col_name = feature + '_' + val
        x_test[col_name] = 0
        x_test.loc[x_test[x_test[feature] == val].index, col_name] = 1
    other_col = feature + '_' + 'Other'
    x_test[other_col] = 1
    x_test.loc[x_test[x_test[feature].isin(top_20)].index, other_col] = 0
    x_test.drop([feature], axis=1, inplace=True)

    return x_test


def generate_baseline(y_train, y_test):
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

    Outputs:
        y_predict (dataframe): the baseline outcome prediction
        y_predict_probability (int): probability that the outcome is 1
    '''
    val = int(y_train.mode())
    y_predict = y_test.copy()
    y_predict.values[:] = val
    y_predict_probability = y_test.copy()
    if val == 1:
        y_predict_probability.values[:] = 1
    else:
        y_predict_probability.values[:] = 0

    return y_predict, y_predict_probability


def temporal_validation(df, date_field=c.DATE_FIELD, target=c.TARGET,
                        start_date=c.START_DATE, end_date=c.END_DATE,
                        gap_days=c.GAP_DAYS,
                        rolling_months=c.ROLLING_MONTHS):
    '''
    Creates testing and training dataframes using a temporal holdouts
    methodology. Given a specified start date for the training dataset and end
    date for the full dataset, as well as a specified gap of days between
    training and testing, divides the original dataset accordingly.
    A rolling window (ie. 6 months for testing) can be established by
    passing this function through a for-loop using the gap days, rolling
    months, and holdouts parameters.

    Inputs:
        df (dataframe): clean dataframe
        date_field (str): name of the date feature on which to split
        target (str): outcome variable name
        start_date (datetime): start date for training data
        end_date (datetime): end date in full data used to determine
                             validation set boundaries
        gap_days (int): number of days for outcome to take effect
        rolling_months (int): rolling window in months for test set

    Outputs:
        x_train (dataframe): training feature dataset
        x_test (dataframe): testing feature dataset
        y_train (dataframe): training outcome dataset
        y_test (dataframe): testing outcome dataset
        start_test (datetime): start of testing set (for documentation)
        end_test (datetime): end of testing set (for documentation)
    '''
    end_test = end_date
    start_test = end_test - relativedelta.relativedelta(months=rolling_months)\
                 + timedelta(days=1) # avoids double counting days
    end_train = start_test - timedelta(days=gap_days) # ie.60 days
    start_train = start_date

    train = df[(df[date_field] >= start_train) & (df[date_field] <= end_train)]
    test = df[(df[date_field] >= start_test) & (df[date_field] <= end_test)]

    features = df.iloc[0].index.to_list()
    features.remove(target)
    features.remove(date_field)
    x_train = train[features]
    y_train = train[[target]]
    y_train = y_train.squeeze()
    x_test = test[features]
    y_test = test[[target]]
    y_test = y_test.squeeze()

    return x_train, x_test, y_train, y_test, start_test, end_test


def preds_at_k(y_pred_probs_sorted, y_test_sorted, k):
    '''
    Given a set of ranked scores and true outcomes from a test dataset,
    along with a value for k (eg. 20), determines a population cutoff
    at k% and classifies the top k% of scores as 1 and the rest as 0. Returns
    this newly classified data along with the associated true outcome for each
    record. Note: for the purpose of this exercise, does not specify tie
    conditions.

    Inputs:
        y_pred_probs_sorted (tuple): sorted tuple of scores for each record
        y_test_sorted (tuple): tuple of true outcomes for each sorted record
        k (float): percentage of population

    Outputs:
        y_true (numpy array): true outcomes
        preds (lst): classified data
    '''
    sorted_scores = np.array(y_pred_probs_sorted)
    idx = np.argsort(sorted_scores)[::-1]
    sorted_trues = np.array(y_test_sorted)
    y_scores = sorted_scores[idx]
    y_true = sorted_trues[idx]
    cutoff_index = int(len(y_scores) * (k / 100.0))
    preds = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]

    return y_true, preds


def plot_precision_recall(y_true, y_prob, model_name, filename):
    '''
    Plots precision-recall curves for a specified model.

    Inputs:
        y_true (series): true test outcomes
        y_prob (numpy array): predicted test outcomes
        model_name (model): a model to plot
        filename (str): filename to save each plot

    Output: Saves a PNG locally plotting the precision-recall curves.
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = \
        precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.figure()
    sns.set(font_scale=0.75)
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    plt.title(model_name)
    plt.savefig(filename, dpi=400)
    plt.cla()
    plt.close(fig)


def run_models(x_train, x_test, y_train, y_test, dates, results,
               models_to_run=c.ANALYSIS, models=c.MODELS, grid=c.SMALL_GRID,
               k_values=c.K_VALUES, create_plot=False, counter=2):
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
        k_values (lst): list of k-values to calculate evaluation metrics
        create_plot (boolean): Prints precision recall curves if true
        counter (int): row number to track precision-recall curves

    Output:
        results (dataframe): the final table summarizing results and metrics
    '''
    # Prepare table
    metrics = ['accuracy_score_{}', 'f1_score_{}', 'precision_{}', 'recall_{}']
    lst = []
    for k in k_values:
        for score in metrics:
            label = score.format(k)
            lst.append(label)
    columns = ('test_dates', 'model_type', 'model', 'train_data_size',
               'test_data_size', 'parameters', 'AUC') + tuple(lst)
    if results.empty:
        results = pd.DataFrame(columns=columns)
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    train_data_size = len(x_train)
    test_data_size = len(x_test)


    # Generate Baseline
    y_predict, y_predict_probability = generate_baseline(y_train, y_test)
    row_base = [dates, 'baseline', 'most_frequent_train_label',\
                train_data_size, test_data_size, 'N/A']
    row_base.append(roc_auc_score(y_test, y_predict_probability))
    for k in k_values:
        row_base.append(accuracy_score(y_test, y_predict))
        row_base.append(f1_score(y_test, y_predict))
        row_base.append(precision_score(y_test, y_predict))
        row_base.append(recall_score(y_test, y_predict))
    results.loc[len(results)] = row_base


    # Generate models
    count = counter
    for index, model in enumerate([models[x] for x in models_to_run]):
        parameters = grid[models_to_run[index]]
        for p in ParameterGrid(parameters):
            try:
                model.set_params(**p)
                trained_model = model.fit(x_train, y_train)
                if models_to_run[index] == 'SVM':
                    y_pred_probs = trained_model.decision_function(x_test)
                else:
                    y_pred_probs = trained_model.predict_proba(x_test)[:, 1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs,
                                                                     y_test.values),
                                                                 reverse=True))
                row = [dates, models_to_run[index], model,\
                       train_data_size, test_data_size, p]
                row.append(roc_auc_score(y_test, y_pred_probs))
                for k in k_values:
                    y_true, k_predict = preds_at_k(y_pred_probs_sorted,
                                                   y_test_sorted, k)
                    row.append(accuracy_score(y_true, k_predict))
                    row.append(f1_score(y_true, k_predict))
                    row.append(precision_score(y_true, k_predict))
                    row.append(recall_score(y_true, k_predict))
                results.loc[len(results)] = row
                if create_plot:
                    plt.rcParams.update({'figure.max_open_warning': 0})
                    plot_precision_recall(y_test, y_pred_probs, model,
                                          'precision_recall_row_' +
                                          str(count) + '.png')
                    count += 1
            except IndexError as e:
                print('Error', e)
                continue

    return results, count
