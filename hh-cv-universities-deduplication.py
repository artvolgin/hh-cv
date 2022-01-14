# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:08:40 2021

@author: Artem
"""

# =============================================================================
# 0. Libraries
# =============================================================================

import pandas as pd
import numpy as np
import os
import getpass
import requests
import re
import urllib
import time
from bs4 import BeautifulSoup

# =============================================================================
# 1. Data Loading
# =============================================================================

# Read the data
os.chdir("/data")
df_cv_edu = pd.read_csv("hh_cv_educ.csv")

# =============================================================================
# 2. Preprocessing
# =============================================================================

# Remove CVs with 5 and more educations
edu_number = df_cv_edu['index1'].value_counts()
keep_id = set(edu_number[edu_number < 5].index)
df_cv_edu = df_cv_edu[df_cv_edu['index1'].apply(lambda i: i in keep_id)]
df_cv_edu['index1'] = df_cv_edu['index1'].astype(str)

# Remove CV with an error
df_cv_edu = df_cv_edu[df_cv_edu['index1'] != 'Магистральный железнодорожный транспорт']

# Remove 1 missing
df_cv_edu = df_cv_edu[df_cv_edu['end_date'].notna()] 

# Select only ending dates from 1950 to 2030
df_cv_edu = df_cv_edu[df_cv_edu['end_date'] > "1950-01-01 00:00:00.000"]
df_cv_edu = df_cv_edu[df_cv_edu['end_date'] < "2030-01-01 00:00:00.000"]

# Transform to pandas datetime
df_cv_edu['end_date'] = pd.to_datetime(df_cv_edu['end_date'])

# Fix time zone problem by adding one day. Take only year from the date
df_cv_edu['end_date'] = (df_cv_edu['end_date'] + pd.DateOffset(1)).apply(lambda t: t.year)


# =============================================================================
# 3. Deduplication of university names
# =============================================================================

# ----- 3.1 Preprocessing of university names
# Remove ""
df_cv_edu['university_name'] = df_cv_edu['university_name'].apply(lambda name: name.replace('"', ''))
# Remove white space
df_cv_edu['university_name'] = df_cv_edu['university_name'].apply(lambda name: name.strip())
# Remove universities with < 3 characters
df_cv_edu = df_cv_edu[df_cv_edu['university_name'].apply(len) >= 3]

# ----- 3.2 Unique university names
table_uninames = pd.DataFrame(df_cv_edu['university_name'].value_counts())
table_uninames.reset_index(inplace=True)
table_uninames.columns = ['name', 'freq']

print((table_uninames['freq'] >= 10).sum()) # Number of universities with freq >= 10
print(table_uninames[table_uninames['freq'] >= 10]['freq'].sum() / table_uninames['freq'].sum()) # Proportional to all occurrences 

# Select universities with frequency >= 10
table_uninames = table_uninames[(table_uninames['freq'] >= 10)]


# ----- 3.3 Geocaching of university names

GOOGLE_API_KEY = 'GOOGLE_API_KEY'

def parseCoordinatesGoogle(address_or_zipcode):
   lat, lng = None, None
   api_key = GOOGLE_API_KEY
   base_url = "https://maps.googleapis.com/maps/api/geocode/json"
   endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
   r = requests.get(endpoint)
   if r.status_code not in range(200, 299):
       return None, None
   try:
       results = r.json()['results'][0]
       lat = results['geometry']['location']['lat']
       lng = results['geometry']['location']['lng']
   except:
       pass
   return lat, lng


# Parse coordinates in a loop
start_time = time.time()
parsed_coordinates = []
for i in range(len(table_uninames)):
    parsed_coordinates.append(parseCoordinatesGoogle(table_uninames['name'].iloc[i]))
    time.sleep(0.3)
    print(i)
print("--- %s seconds ---" % (time.time() - start_time))
pd.Series(parsed_coordinates).to_pickle("parsed_coordinates.obj")
parsed_coordinates = pd.read_pickle("parsed_coordinates.obj")

# Add coordinates to the dataframe
table_uninames['coords'] = list(parsed_coordinates)
table_uninames['coords'] = table_uninames['coords'].apply(lambda x: str(x[0]) + "_" + str(x[1]))

# Parse missing coordinates
table_uninames_parsed = table_uninames[table_uninames['coords'] != "None_None"]
table_uninames_notparsed = table_uninames[table_uninames['coords'] == "None_None"]

start_time = time.time()
parsed_coordinates_2 = []
for i in range(len(table_uninames_notparsed)):
    parsed_coordinates_2.append(parseCoordinatesGoogle(table_uninames_notparsed['name'].iloc[i]))
    time.sleep(1)
    print(i)
print("--- %s seconds ---" % (time.time() - start_time))

table_uninames_notparsed['coords'] = parsed_coordinates_2
table_uninames_notparsed['coords'] = table_uninames_notparsed['coords'].apply(lambda x: str(x[0]) + "_" + str(x[1]))

# Combine two dataframes
table_uninames = pd.concat([table_uninames_parsed, table_uninames_notparsed])
# Save to pickle
table_uninames.to_pickle("uninames_coordinates.obj")


# ----- 3.4 Deduplication based on the coordinates
table_uninames = pd.read_pickle("uninames_coordinates.obj")

# Remove universities without coordinates
table_uninames = table_uninames[table_uninames['coords'] != "None_None"]

# Create dictionary with key:coords, value:most popular name
table_uninames = table_uninames.sort_values("freq", ascending=False)
dict_coords_popname = dict(zip(table_uninames.drop_duplicates("coords")['coords'],
                               table_uninames.drop_duplicates("coords")['name']))

# Replace names in table with most popular name for each university
table_uninames['popular_name'] = table_uninames['coords'].apply(lambda coord: dict_coords_popname[coord])

# Recode name to the most popular name
dict_name_popular = dict(zip(list(table_uninames['name']),
                             list(table_uninames['popular_name'])))
df_cv_edu['university_name'] = df_cv_edu['university_name'].apply(
    lambda name: dict_name_popular[name] if name in dict_name_popular else np.nan)
df_cv_edu = df_cv_edu[df_cv_edu['university_name'].notna()]

# Aggregate frequency by the most popular name
table_uninames_agg = pd.DataFrame(table_uninames.groupby('popular_name')['freq'].sum()).reset_index()

# Save to excel for coding
# table_uninames_agg.to_excel("uninames_agg.xlsx")


# ----- 3.5 Check for the presence of the most popular university names
table_uninames_agg = pd.read_excel("uninames_agg.xlsx")
table_uninames_hh_busgov = pd.read_excel("uninames_hh_busgov.xlsx")
hh_names = set(np.concatenate(table_uninames_hh_busgov.loc[:, ['name_hh_1', 'name_hh_2', 'name_hh_3', 'name_hh_4', 'name_hh_5']].values))
hh_names.remove(np.nan)
table_uninames_agg['hh_busgov'] = table_uninames_agg['popular_name'].apply(lambda name: int(name in hh_names))


# ----- 3.6 Recode university names from hh to busgov
table_uninames_hh_busgov = pd.read_excel("uninames_hh_busgov.xlsx")
table_uninames_hh_busgov = table_uninames_hh_busgov.loc[:, table_uninames_hh_busgov.columns != 'region_busgov']

# Create dictionary
dict_names_hh_busgov = pd.melt(table_uninames_hh_busgov, id_vars=['name_busgov'],
                               value_vars=table_uninames_hh_busgov.columns[table_uninames_hh_busgov.columns != 'name_busgov'])
dict_names_hh_busgov = dict_names_hh_busgov[dict_names_hh_busgov['value'].notna()]
dict_names_hh_busgov = dict(zip(
    list(dict_names_hh_busgov['value']),
    list(dict_names_hh_busgov['name_busgov'])))

# Recode university names from hh to busgov
df_cv_edu['university_name'] = df_cv_edu['university_name'].apply(lambda name: name.replace('"', ''))
df_cv_edu['university_name_busgov'] = df_cv_edu['university_name'].apply(
    lambda name: dict_names_hh_busgov[name] if name in dict_names_hh_busgov else np.nan)

# Proportion of CVs with deduplicated name to all CVs
df_cv_edu['university_name_busgov'].notna().sum() / len(df_cv_edu)

# Select only CVs with deduplicated name
df_cv_edu = df_cv_edu[df_cv_edu['university_name_busgov'].notna()]

# Save to pickle
df_cv_edu.to_pickle("df_cv_edu.obj")
