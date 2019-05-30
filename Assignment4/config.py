'''
Config File - Machine Learning Pipeline

Use this file to set parameters for ml_pipeline.py and analysis.py. These
variables are currently set to the Donorschoose dataset but can be easily
modified to conduct an analysis on a new dataset. Alternatively, values can
be passed directly into each function as parameters.

Author: Tammy Glazer
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
                             BaggingClassifier, GradientBoostingClassifier


### Data preparation, exploration, and processing variables ###

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
BINS = 3
# BINS = pd.IntervalIndex.from_tuples([(20, 39), (40, 69), (70, 110)],
                                     #closed='both')
# Specify labels for discrete bins or leave list empty for default label
LABEL = []
# Specify a feature to create dummies
DUMMIES = 'grade_level'
# Specify test size and random state if not using temporal holdouts methodology
# TEST_SIZE = 0.2
# RANDOM_STATE = 1



### File names to print tables/graphs/plots to ###

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



### Analysis parameters ###

# Specify columns that are categorical=FEW (with <20 categories), categorical=
# MANY (with 20+ categories), continuous, and columns to drop during analysis
CATEGORICAL_VARS_FEW = ['school_metro', 'school_charter', 'school_magnet',\
                        'teacher_prefix', 'poverty_level', 'grade_level',\
                        'eligible_double_your_impact_match']
CATEGORICAL_VARS_MANY = ['school_city', 'school_state', 'school_district',\
                         'school_county', 'primary_focus_subject',\
                         'primary_focus_area', 'secondary_focus_subject',\
                         'secondary_focus_area', 'resource_type']
# Note: pipeline uses MinMaxScaler to scale continuous variables
CONTINUOUS_VARS = ['total_price_including_optional_support', 'students_reached']
DROP = ['teacher_acctid', 'schoolid', 'school_ncesid', 'school_latitude',\
        'school_longitude', 'datefullyfunded']
# Specify k values for evaluation metrics (if score >= k%, then 1)
K_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]
# Specify fields for temporal holdouts validation
DATE_FIELD = 'date_posted'
START_DATE = pd.to_datetime('2012-01-01 00:00:00')
END_DATE = pd.to_datetime('2013-12-31 00:00:00')
GAP_DAYS = 60
ROLLING_MONTHS = 6
HOLDOUTS = 3
# Specify models to run
ANALYSIS = ['LR', 'KNN', 'DT', 'SVM', 'RF', 'BOOST', 'GRADIENT', 'BAG']

# Define parameters for classifiers
MODELS = {'LR': LogisticRegression(penalty='l1',
                                   C=0.00001,
                                   class_weight=None,
                                   random_state=1),
          'KNN': KNeighborsClassifier(n_neighbors=3,
                                      weights='uniform',
                                      algorithm='auto',
                                      p=2),
          'DT': DecisionTreeClassifier(criterion='gini',
                                       max_depth=5,
                                       min_samples_split=2,
                                       min_samples_leaf=1),
          'SVM': LinearSVC(C=0.00001,
                           loss='hinge',
                           random_state=1),
          'RF': RandomForestClassifier(n_estimators=5,
                                       max_depth=5,
                                       min_samples_split=2,
                                       min_samples_leaf=1),
          'BOOST': AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                                      algorithm='SAMME.R',
                                      n_estimators=10,
                                      random_state=1),
          'GRADIENT': GradientBoostingClassifier(learning_rate=0.5,
                                                 subsample=0.5,
                                                 max_depth=5,
                                                 n_estimators=100),
          'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=10),
                                   n_estimators=10,
                                   random_state=1)}

SMALL_GRID = {'LR': {'penalty': ['l1', 'l2'],
                     'C': [0.00001, 1],
                     'class_weight': [None],
                     'random_state': [1]},
              'KNN': {'n_neighbors': [1, 5],
                      'weights': ['uniform'],
                      'algorithm': ['auto'],
                      'p': [1, 2]},
              'DT': {'criterion': ['gini'],
                     'max_depth': [1, 5],
                     'min_samples_split': [2, 10],
                     'min_samples_leaf': [1]},
              'SVM': {'C': [0.00001, 1],
                      'loss': ['hinge'],
                      'random_state': [1]},
              'RF': {'n_estimators': [10],
                     'max_depth': [1, 5],
                     'min_samples_split': [2, 10],
                     'min_samples_leaf': [1]},
              'BOOST': {'n_estimators': [1, 10],
                        'algorithm': ['SAMME.R'],
                        'random_state': [1]},
              'GRADIENT': {'learning_rate': [0.001, 0.5],
                           'subsample': [0.5],
                           'max_depth': [5],
                           'n_estimators': [10, 100]},
              'BAG': {'n_estimators': [1, 10],
                      'random_state': [1]}}

LARGE_GRID = {'LR': {'penalty': ['l1', 'l2'],
                     'C': [0.00001, 0.001, 0.1, 1, 10],
                     'class_weight': [None, 'balanced'],
                     'random_state': [1]},
              'KNN': {'n_neighbors': [1, 5, 10, 25, 100],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                      'p': [1, 2]},
              'DT': {'criterion': ['gini', 'entropy'],
                     'max_depth': [1, 5, 10, 50, 100],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 5]},
              'SVM': {'C': [0.00001, 0.001, 0.1, 1, 10],
                      'loss': ['hinge', 'squared_hinge'],
                      'random_state': [1]},
              'RF': {'n_estimators': [1, 10, 100, 1000, 10000],
                     'max_depth': [1, 5, 10, 50, 100],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 5]},
              'BOOST': {'n_estimators': [1, 10, 100, 1000],
                        'algorithm': ['SAMME', 'SAMME.R'],
                        'random_state': [1]},
              'GRADIENT': {'learning_rate': [0.001, 0.01, 0.1, 0.5],
                           'subsample': [0.1, 0.5, 1.0],
                           'max_depth': [1, 3, 5, 20, 100],
                           'n_estimators': [1, 10, 100, 10000]},
              'BAG': {'n_estimators': [1, 10, 100, 1000, 10000],
                      'random_state': [1]}}
