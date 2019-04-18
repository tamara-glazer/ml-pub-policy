'''
Machine Learning Pipeline
1. Read/Load Data (CSV)
2. Explore Data (print statements, tables, and graphs)
3. Pre-Process and Clean Data (adjust missing values)
4. Generate Features/Predictors (discretizes and creates dummies)
5. Build Machine Learning Classifier (Decision Tree class)
6. Evaluate Classifier (Accuracy score)

Author: Tammy Glazer
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Raw data CSV filename
RAW_DATA = 'credit-data.csv'
# Name of column to use as Index (specify '' if N/A)
UNIQUE_ID = 'PersonID'
# Specify of a single variable to view its distribution as a histogram
DISTRIBUTION = 'age'
# List two continuous variables to view a scatterplot (examine correlation)
TWO_CONTINUOUS = ['age', 'DebtRatio']
# True if outcome variable is binary, otherwise state False
BINARY = True
# State the name of the outcome variable (specify '' if N/A)
TARGET = 'SeriousDlqin2yrs'
# List any two categorical variables to view a heatmap (examine correlation)
TWO_CATEGORICAL = ['NumberOfDependents', 'NumberRealEstateLoansOrLines']
# Specify a continuous and categorical variable to view a barplot
CATEGORICAL_VAR = 'NumberOfDependents'
CONTINUOUS_VAR = 'DebtRatio'
# Specify a NUMERIC attribute to print a list of outliers for that attribute
OUTLIER = 'age'
# Specify a continuous feature to discretize
FEATURE = 'age'
# Specify bins as inclusive integer tuples to discretize the above variable.
# Alternatively, BINS can be set to an integer. Comment out option not is use.
BINS = pd.IntervalIndex.from_tuples([(20, 39), (40, 69), (70, 110)],
                                    closed='both')
#BINS = 5
# Specify labels for discrete bins or leave list empty for default label
LABEL = []
# Specify a feature to create dummies from
DUMMIES = 'zipcode'
# Specify test size (train-test-split) and random state (seed)
TEST_SIZE = 0.2
RANDOM_STATE = 1


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


def read_data(file=RAW_DATA):
    '''
    Reads in a CSV using pandas and converts the CSV into a pandas dataframe.
    Sets the index to a unique identifier if an attribute(s) is specified.

    Input:
        file (str): filename for a CSV

    Output:
        df (dataframe): a pandas dataframe
    '''
    df = pd.read_csv(file)
    if UNIQUE_ID:
        df.set_index(UNIQUE_ID, inplace=True)
    
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
            print('The index represent unique entities')
        else:
            print('The index DOES NOT represent unique entities')

    df.dtypes.to_csv(DATATYPES, header=False)

    df.describe().to_csv(RANGE, header=False)
    
    nulls = df.isna().sum().to_frame('count_null')
    nulls['pct_null'] = (nulls['count_null'] / df.size).round(4)
    nulls.to_csv(NULLS)

    df.corr().round(4).to_csv(CORRELATION_TABLE)
    
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
    plt.savefig(CORRELATION_IMAGE, dpi=400)
    plt.close()


def distribution(df, distribution=DISTRIBUTION, file_1=HISTOGRAM):
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
    sns.distplot(df[distribution].dropna(), hist=True,
                                        kde=False,
                                        color = 'blue',
                                        hist_kws={'edgecolor':'black'})
    plt.title('{} distribution'.format(distribution))
    plt.xlabel(distribution)
    plt.ylabel('Number of Observations')
    plt.savefig(file_1, dpi=400)
    plt.close()


def create_scatterplot(df, comparison=TWO_CONTINUOUS, file_1=SCATTERPLOT):
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


def create_heatmap(df, comparison=TWO_CATEGORICAL, file_1=HEATMAP):
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
    groups = df.groupby(x).agg({y: 'mean'})
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
    For each column, fills in missing values with the mean value for the
    column. Note that this function can be applied to numeric columns only.

    Input:
        df (dataframe): a pandas dataframe

    Output:
        df (dataframe): an updated pandas dataframe
    '''
    for col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

    return df


def discretize_continuous_variable(df, feature=FEATURE, bins=BINS, label=LABEL):
    '''
    Creates a new column in the dataframe containing a categorical variable
    to represent a specified continuous variable. Bins can be set automatically
    or manually.

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

    return df


def create_dummies(df, feature=DUMMIES):
    '''
    Takes a categorical variable and creates dummy variables from it,
    which are concatenated to the end of the dataframe. Dummy headers will
    contain a prefix of up to 3 characters from the original variable
    name

    Inputs:
        df (dataframe): a pandas dataframe
        feature (str): a variable name
    
    Output:
        df (dataframe): dataframe with new dummy variable columns
    '''
    dummies = pd.get_dummies(df[feature], prefix=feature[:3])
    df = pd.concat([df, dummies], axis=1)

    return df


class Classifier:
    '''
    Class for representing a decision tree classifier

    Attributes:
        x_data (dataframe): a dataframe containing all features
        y_data (dataframe): a dataframe containing the target variable
        x_train (dataframe): a dataframe containing the training features
        y_train (dataframe): a dataframe containing the training target
        x_test (dataframe): a dataframe containing the testing features
        y_test (dataframe): a dataframe containing the testing target
        trained_model (obj): decision tree trained on the training data
        y_hat (array): an array of predicted outcomes for the x_test dataframe
        accuracy (float): the prediction accuracy score
        predictor_set_size (int): number of features used to predict
    '''
    def __init__(self, df):
        self.x_data = self.create_x(df)
        self.y_data = self.create_y(df)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.trained_model = None
        self.train()
        self.y_hat = self.predict()


    def create_x(self, df):
        '''
        Creates a dataframe of all features to be used as predictors

        Input:
            df (dataframe): cleaned dataframe (containing categorical vars)

        Output:
            df (dataframe): features dataframe
        '''
        features = df.iloc[0].index.to_list()
        features.remove(TARGET)
        
        return df[features]


    def create_y(self, df):
        '''
        Creates a dataframe of the full outcome column to be used as target

        Input:
            df (dataframe): cleaned dataframe (containing categorical vars)

        Output:
            df (dataframe): target dataframe
        '''
        return df[[TARGET]]


    def train(self, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        '''
        Splits the full x and y data into training and testing sets based on
        a specified testing size and random seed. Trains a Decision Tree
        model to predict y values based on given x values.

        Inputs:
            test_size (float): the proportional size of the testing data
            random_state (int): random seed
        '''
        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(self.x_data, self.y_data, test_size=TEST_SIZE, \
                         random_state=RANDOM_STATE)
        model = tree.DecisionTreeClassifier()
        self.trained_model = model.fit(self.x_train, self.y_train)


    def predict(self):
        '''
        Runs a prediciton on the trained model from the testing data
        '''
        y_hat = self.trained_model.predict(self.x_test)
        
        return y_hat


    def visualize(self, file_1=TREE):
        '''
        Exports a visualization of the trained tree

        Inputs:
            file_1 (str): filename to export image
        '''
        tree.export_graphviz(self.trained_model, out_file=file_1)


    @property
    def accuracy(self):
        '''
        Reports the accuracy of the trained classifier based on testing data
        
        Output:
            accuracy_score (float): accuracy score
        '''
        return accuracy_score(self.y_test, self.y_hat)


    @property
    def predictor_set_size(self):
        '''
        Reports size of the predictor variable set

        Output:
            int: predictor set size
        '''
        return len(self.x_train.columns)


def buildtree(df):
    '''
    Creates and traings a decision tree using the Classifier class

    Input:
        df (dataframe): clean dataframe (categorical variables only)
    
    Output:
        tree (obj): returns a trained class object
    '''
    tree = Classifier(df)

    return tree
