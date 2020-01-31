# Final Machine Learning Pipeline (See: Assignment 4)
Leverages a modular, extensible, machine learning pipeline to train a variety
of models to predict if projects on DonorsChoose will NOT get fully funded
within 60 days of posting. Calculates evaluation metrics to better understand
the strength of each model and validate assumptions.

## Purpose
The purpose of the file ml_pipeline.py us to serve as a simple, extensible,
machine learning pipeline that can read/load data, explore data through
a variety of summary tables and visualizations, pre-process and clean data,
generate features/predictors, build machine learning classifiers, and
provide several evaluation metrics. This pipeline leverages scikit-learn's
LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, LinearSVC,
RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, and
BaggingClassifier modules to build classification models. The purpose of the
file analysis.py is to use the machine learning pipeline to load and
pre-process the DonorsChoose dataset and train a variety of machine learning
models to predict if projects will not get fully funded within 60 days of
posting with a measured degree of accuracy, precision, recall, AUC, and F1.
The purpose of the file config.py is to set parameters for ml_pipeline.py
and analysis.py. Training and testing datasets are created using a rolling
window of 6 months, providing 3 test sets, with 60 days between training and
testing sets.

1. Read/Load Data (CSV)
2. Explore Data
3. Pre-Process and Clean Data
4. Generate Features/Predictors
5. Build Machine Learning Classifiers
6. Evaluate Classifiers

## Libraries
These scripts leverage the following Python libraries:

```python
from datetime import timedelta
from dateutil import relativedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
                             BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score,\
							recall_score, f1_score, precision_recall_curve
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
```

## Data and Files
The dataset used for this exercise is a modified version of the
[Kaggle DonorsChoose Dataset](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data)

A CSV is included containing this modified dataset for easy reference
(projects_2012_2013.csv). The extensible machine_learning pipeline is
in the file: ml_pipeline.py. This pipeline is applied to the DonorsChoose
Dataset using a second file: analysis.py. Global parameters are set using the
file config.py.

The final writeup is included as a PDF in the repository (glazer_writeup.pdf).
This file was also submitted on Canvas.

## Usage
The final script can be run from the command line or from within ipython3.
As the script runs, tables will be exported to the current directory as CSV
and PNG files. By default, the scripts run the entire pipeline on an
abbreviated parameter grid and do not save precision-recall plots locally. To
override these default settings, simply pass 'plot=True' or 'grid=c.LARGE_GRID'
into a.run_pipeline().


Command line:

```bash
python3 analysis.py
```

ipython3:

```python
import analysis as a

a.run_pipeline()
```

For the machine learning pipeline to be leveraged successfully with future
datasets, all global variables in config.py file can be updated based on the
conditions and instructions provided. These will update all necessary
parameters in the other two files. Alternatively, parameters can be directly
passed into functions to override default values.
