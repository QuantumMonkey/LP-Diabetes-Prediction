"""
Phase 1: Data Visualization

This data is information about the shows and movies streaming in the USA.
The intention here is to find out how Netflix content is being consumed by the audience.
"""

import numpy as np  # to perform linear algebra
import pandas as pd  # to perform data and file processing
import matplotlib.pyplot as plt  # for plotting data
import seaborn as sns  # for visualizing data

# Import data into DataFrames
titles_data = pd.read_csv('Dataset/titles.csv')
credits_data = pd.read_csv('Dataset/credits.csv')

# Data Cleaning
# Titles
print("Top 5 rows:\n", titles_data.head(), '\n')  # Display top 5 rows
print("Dimensions: ", titles_data.shape, '\n')  # Data Structure
print("Statistical details:\n", titles_data.describe(), '\n')  # Display statistical details of the dataset
print("Schema:\n", titles_data.info(), '\n')  # Display data information and datatypes
print("Duplicate records: ", titles_data.duplicated().sum(), '\n')  # Check for duplicate records
print("Null values in columns:\n", titles_data.isnull().sum(), '\n')  # Display number of null values in columns

# Drop non-essential columns
titles_data.drop(['description', 'age_certification', 'seasons', 'imdb_id', 'imdb_votes'], axis=1, inplace=True)
print("Top 5 rows:\n", titles_data.head(), '\n')  # Display top 5 rows

titles_data.dropna(axis=0, how='any', inplace=True)
print("Null values in columns:\n", titles_data.isnull().sum(), '\n')  # Display number of null values in columns
print("Dimensions: ", titles_data.shape, '\n')  # Data Structure

# Credits
print("Top 5 rows:\n", credits_data.head(), '\n')  # Display top 5 rows
print("Dimensions: ", credits_data.shape, '\n')  # Data Structure
print("Statistical details:\n", credits_data.describe(), '\n')  # Display statistical details of the dataset
print("Schema:\n", credits_data.info(), '\n')  # Display data information and datatypes
print("Duplicate records: ", credits_data.duplicated().sum(), '\n')  # Check for duplicate records
print("Null values in columns:\n", credits_data.isnull().sum(), '\n')  # Display number of null values in columns

# Drop non-essential columns
credits_data.drop(['character', 'person_id'], axis=1, inplace=True)
print("Top 5 rows:\n", credits_data.head(), '\n')  # Display top 5 rows

credits_data.dropna(axis=0, how='any', inplace=True)
print("Null values in columns:\n", credits_data.isnull().sum(), '\n')  # Display number of null values in columns
print("Dimensions: ", credits_data.shape, '\n')  # Data Structure

# Exploratory Data Analysis - time to plot some insights

# Top 10 Countries to produce content
top10_production_countries = titles_data.production_countries.value_counts().head(10)
top10_production_countries.plot(kind='bar', width=0.8, figsize=(9, 6), color='r')  # plot a bar graph
plt.title('Top 10 countries with highest production')  # Title
plt.xlabel('Countries')  # X-axis label
plt.ylabel('Production Count')  # Y-axis label
plt.show()

# Content production per year
year_count = titles_data.value_counts()
print("Movies produced in the last 10 years:\n", year_count.head(10))

sns.lineplot(data=year_count)  # Feed data to seaborn graph
plt.title('Total shows/movies released over the years')  # Title
plt.xlim(1950, 2030)  # Limit range for X-axis
plt.xlabel('release year')  # X-axis label
plt.ylabel('total')  # Y-axis label
plt.show()

# Types of content on Nwtflix
type_count = titles_data.type.value_count()
print("Types of content with count:\n", type_count.head())

type_count.plot('pie', figsize=(10,5), autopct='%1.1f%%')
