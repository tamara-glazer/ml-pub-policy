'''
DESCRIPTION

Author: Tammy Glazer
'''

import pandas as pd
from sodapy import Socrata

URL = 'data.cityofchicago.org'
ENDPOINT = '6zsd-86xi'
TOKEN = 'YNQSgf5W1B65zdCeAX4nY9CXl'


def load_data(url=URL, token=TOKEN, endpoint=ENDPOINT):
	'''
	Uses the Socrata Open Data API (SODA) to retrieve data from the Chicago
	Data Portal on crimes reported between 2017 and 2018 using a unique,
	public app token. Converts JSON retrieved from API to a list of
	dictonaries and then to a pandas DataFrame.

	Inputs:
		url (str): url for the City of Chicago Open Data Portal
		token (str): public app token created for "tglazer_assignment1"
		endpoint (str): endpoint details to direct the API

	Output:
		df (dataframe): a pandas dataframe containing data for 2017 and 2018
	'''
	client = Socrata(URL, TOKEN)
	results = client.get(ENDPOINT, where='year >= 2017 and year <=2018')
	df = pd.DataFrame.from_records(results)

	return df
