# Intro
"""
This data is information about the shows and movies streaming in the USA by Netflix.
We are going to try to replicate the given ratings by predicting them with the given data as accurately as possible.

"""
# Packages
import numpy as np  # To perform linear algebra
import pandas as pd  # To perform data and file processing
import matplotlib.pyplot as plt  # For plotting data
import seaborn as sns  # For visualizing data
from sklearn.model_selection import train_test_split, cross_val_score  # Train Test Split and Average RMSE # calculation
from sklearn.tree import DecisionTreeRegressor  # Decision Tree Algorithm

# Methods
def data_info(dataset):  # Display stats and null values
    """
    :param dataset: The table to print statistical information about.
    :return: Displayed statistics and structure.
    """
    print("Top 5 rows:\n", dataset.head(), '\n')  # Display top 5 rows
    print("Dimensions: ", dataset.shape, '\n')  # Data Structure
    print("Statistical details:\n", dataset.describe(), '\n')  # Display statistical details of the dataset
    print("Schema:\n", dataset.info(), '\n')  # Display data information and datatypes
    print("Duplicate records: ", dataset.duplicated().sum(), '\n')  # Check for duplicate records
    print("Null values in columns:\n", dataset.isnull().sum(), '\n')  # Display number of null values in columns


def data_clean(dataset):  # Drop null values
    """
        :param dataset: The table with null values to be removed/cleaned.
        :return: Cleaned dataset with no null values.
    """
    dataset.dropna(axis=0, how='any', inplace=True)
    print("Null values in columns:\n", dataset.isnull().sum(), '\n')  # Display number of null values in columns
    print("Dimensions: ", dataset.shape, '\n')  # Data Structure
    return dataset


# Import data into DataFrames
titles_data = pd.read_csv('Dataset/titles.csv')
credits_data = pd.read_csv('Dataset/credits.csv')

# Data cleaning
data_info(titles_data)
titles_data.drop(['description', 'age_certification', 'seasons', 'imdb_id', 'imdb_votes'], axis=1, inplace=True)
print("Top 5 rows:\n", titles_data.head(), '\n')
titles_data = data_clean(titles_data)
# Checked Titles dataframe, removed non-essential columns as well as purged duplicate and null values.

modelling_titles_data = titles_data  # Backup if the merged data doesn't work out


data_info(credits_data)
credits_data.drop(['character', 'person_id'], axis=1, inplace=True)
print("Top 5 rows:\n", credits_data.head(), '\n')
data_clean(credits_data)
# Checked Credits dataframe, removed non-essential columns as well as purged duplicate and null values.

# Plotting usable raw data EDA
"""
Exploratory Data Analysis - time to plot some insights
"""

# Top 10 Countries to produce content
top10_production_countries = titles_data.production_countries.value_counts().head(10)

top10_production_countries.plot(kind='bar', width=0.8, figsize=(9, 6), color='r')  # plot a bar graph
plt.title('Top 10 countries with highest production')  # Title
plt.xlabel('Countries')  # X-axis label
plt.ylabel('Production Count')  # Y-axis label
# plt.show()

# Content production per year
year_count = titles_data.release_year.value_counts()
print("Content produced in the last 10 years:\n", year_count.head(10))

sns.lineplot(data=year_count)  # Feed data to seaborn line graph
plt.title('Total shows/movies released over the years')  # Title
plt.xlim(1950, 2030)  # Limit range for X-axis
plt.xlabel('release year')  # X-axis label
plt.ylabel('total')  # Y-axis label
# plt.show()

# Types of content on Netflix
type_count = titles_data.type.value_counts()
print("\nTypes of content with count:\n", type_count.head())

type_count.plot(kind='pie', figsize=(10, 5), autopct='%1.1f%%')  # Plot Pie Chart
plt.title('Type Distribution')  # Title
# plt.show()

# Highly voted content

top10_tmdb_rating = titles_data.sort_values(['tmdb_score', 'tmdb_popularity'], ascending=False)[
    ['title', 'tmdb_score', 'tmdb_popularity', 'type']].head(10)
print("Top 10 movies/shows based on ratings:\n", top10_tmdb_rating)

top10_tmdb_rating.plot(kind='barh', x='title', y='tmdb_popularity', figsize=(9, 6),
                       color='green')  # Plot Horizontal Bar graph
plt.title('Top 10 based on tmdb votes')
plt.xlabel('tmdb_popularity')  # X-axis label
plt.ylabel('Title')  # Y-axis label
# plt.show()

# Merging data for top actors and directors
titles_data = titles_data.merge(credits_data, how='outer', on='id')

# Filtering Directors and Actors
director = titles_data[titles_data['role'] == 'DIRECTOR']
actor = titles_data[titles_data['role'] == 'ACTOR']

# Top 10 Directors
top10_directors = director.sort_values(['tmdb_score', 'tmdb_popularity'], ascending=False)[
    ['name', 'tmdb_score', 'tmdb_popularity']].head(10)
print("Top 10 directors based on ratings:\n", top10_directors)

top10_directors.plot(kind='barh', x='name', y='tmdb_popularity', figsize=(9, 6),
                     color='yellow')  # Plot Horizontal Bar graph
plt.title("Top 10 Directors based on tmdb popularity")  # Title
plt.xlabel('tmdb_popularity')  # X-axis label
plt.ylabel('name')  # Y-axis label
# plt.show()

# Top 10 Actors
top10_actors = actor.sort_values(['tmdb_score', 'tmdb_popularity'], ascending=False)[
    ['name', 'tmdb_score', 'tmdb_popularity']].head(10)
print("Top 10 actors based on ratings:\n", top10_actors)

top10_actors.plot(kind='barh', x='name', y='tmdb_popularity', figsize=(9, 6),
                  color='violet')  # Plot Horizontal Bar graph
plt.title("Top 10 Actors based on tmdb popularity")  # Title
plt.xlabel('tmdb_popularity')  # X-axis label
plt.ylabel('name')  # Y-axis label
# plt.show()


# Gaussian plotting
# Standardization/Normalization
# Final cleaning on data
# Create training and testing data
# Build 3 models to determine most accurate with root-mean-square error
# Create output files for each model
# Interpret results with more plots
# Finish report and attach here
