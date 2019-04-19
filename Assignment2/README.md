# Machine Learning Assignment 2 - Machine Learning Pipeline
Leverages a modular, extensible, machine learning pipeline to train a Decision
Tree model to predict who will experience financial distress in the next two
years based on data from Kaggle's Credit Dataset.

## Purpose
The purpose of the file ml_pipeline.py us to serve as a simple, extensible,
machine learning pipeline that can read/load data, explore data through
a variety of summary tables and visualizations, pre-process and clean data,
generate features/predictors, build a machine learning classifier, and
provide an evaluation metric for this classifier. This pipeline leverages
scikit-learn's Decition Tree module to build a Classifier class. The purpose
of the file final_analysis.py is to use the machine learning pipeline to
load and pre-process the Credit Dataset and to train a Decision Tree this data
to predict who will experience financial distress in the next two years
with a measured degree of accuracy.

## Libraries
These scripts leverage the following Python libraries:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

## Data and Files
The dataset used for this exercise is a modified version of the
[Kaggle Credit Dataset](https://www.kaggle.com/c/GiveMeSomeCredit/data)

A CSV is included containing this modified dataset for easy reference
(credit-data.csv). The extensible machine_learning pipeline is in the file:
ml_pipeline.py. This pipeline is applied to the Credit Dataset within a
second file: final_analysis.py.

The final writeup, including all tables and images, is included as a PDF in
the repository (glazer_writeup.pdf). This file was also submitted on Canvas.

## Usage
The final script can be run from the command line or from within ipython3.
As the script runs, several tables will be exported to the current directory
as PNG and Excel files. These tables and images are included directly
in the final writeup.

Command line:

```bash
python3 final_analysis.py
```

ipython3:

```python
import final_analysis as f

tree = f.run_pipeline()
tree.accuracy
```

For the machine learning pipeline to be leveraged successfully with future
datasets, all global variables at the top of the ml_pipeline final can
be updated based on the conditions and instructions provided. These will
directly update all necessary parameters in the remainder of the file.
Alternatively, parameters can be directly passed into functions to override
default values.
