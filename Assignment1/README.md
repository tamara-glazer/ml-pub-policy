# Machine Learning Assignment 1 - Diagnostic
Conducts an analysis of 2017-2018 Crime Reports in Chicago from the Chicago
Open Data Portal joined with American Community Survey data on demographic
characteristics

## Purpose
The purpose of this script is to conduct a thorough analysis of Chicago Crime
Report Data for 2017-2018 and to generate insights into the types of
locations that experience high crime rates. This code can be used to download
Crime Report Data from the Chicago Open Data Portal, generate summary
statistics for crime reports, join in data containing neighborhood names,
generate zip codes based on latitude and longitude fields, join Crime
Report Data with ACS data by zip code, and conduct an analysis on the
augmented dataset. Special attention is given to better understanding crimes
that occur in Garfield Park and Uptown.

## Libraries
This script leverages the following Python libraries:

```python
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from uszipcode import SearchEngine
```

## Data and Files
An API is leveraged to download data from the Chicago Open Data Portal using
the following URL for which the two years of interest can be selected using
the global variable DATA_YEARS (currently set to 2017 and 2018):
[2018 Chicago Crime Report](https://data.cityofchicago.org/resource/6zsd-86xi.json?year=2018&$limit=600000)

An API is also leveraged to download data from the American Community Survey
using the following URL for which a list of variables of interest can be
selected using the global variable DEMOGRAPHICS (currently set to B01003_001E,
B02001_003E, B15003_017E, B08121_001E, and B07012_002E).

A CSV is included containing a list of zip codes in the Chicago Metropolitan
Area (chicago_zip_codes.csv). A CSV is also included that maps community area
numbers to community area neighborhood names (community_area_names.csv). Both
of these files are joined into the final dataframe and must be present in the
directory where the script is run.

The final writeup, including all tables and images, is included as a PDF in
the repository (glazer_writeup.pdf). This file was also submitted on Canvas.

## Usage
The script can be run from the command line or from within ipython3. As the
script runs, several tables will be exported to the current directory as
PNG and Excel files. These tables and images are included directly
in the final writeup.

Command line:

```bash
python3 assignment_1.py
```

ipython3:

```python
import assignment_1 as a

a.full_analysis()
```

Note that the following call within the full_analysis() function takes about
14 minutes to run, as it leverages the uszipcode library to generate zip
codes for each unique block in the Crime Report Dataset. The entire function
runs in approximately 16 minutes.

```python
crime_df = create_crime_zip_codes(df)
```
