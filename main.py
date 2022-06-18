"""
Phase 1: Data Visualization

This data is information about the shows and movies streaming in the USA.
The intention here is to find out how Netflix content is being consumed by the audience.
"""

import numpy as np  # to perform linear algebra
import pandas as pd  # to perform data and file processing
import matplotlib.pyplot as plt  # for plotting data
import seaborn as sns  # for visualizing data


def data_info(dataset):  # Display stats and null values
    """
    :param dataset: The table to print statistical information about.
    :return: Displayed statistics and structure
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


def data_format(dataset, column):
    """

    :param dataset: The table to perform SQL operations on. Used to search unique records to factorize.
    :param column: The column we need to factorize
    :return: Nothing. The column will be factorized and typecast-ed in the dataframe directly.
    """
    from pandasql import sqldf

    pysqldf = lambda q: sqldf(q, globals())

    unique_values_query = "select distinct {} from {}".format(column, dataset)  # Find unique values
    unique_values_df = pysqldf(unique_values_query)  # Run PandaSQL command
    print(unique_values_df)  # Result type is dataframe

    unique_values_list = unique_values_df[column].tolist()  # Convert to list

    for seek in len(unique_values_list):  # Traverse list length
        unique_values_list[seek] = seek  # Replace list item with index (numerical)


# Import data into DataFrames
titles_data = pd.read_csv('Dataset/titles.csv')
credits_data = pd.read_csv('Dataset/credits.csv')

# Data Cleaning
# Titles
data_info(titles_data)

# Drop non-essential columns
titles_data.drop(['description', 'age_certification', 'seasons', 'imdb_id', 'imdb_votes'], axis=1, inplace=True)
print("Top 5 rows:\n", titles_data.head(), '\n')  # Display top 5 rows

titles_data = data_clean(titles_data)

modelling_titles_data = titles_data  # Backup if the merged data doesn't work out

# Credits
data_info(credits_data)

# Drop non-essential columns
credits_data.drop(['character', 'person_id'], axis=1, inplace=True)
print("Top 5 rows:\n", credits_data.head(), '\n')  # Display top 5 rows

data_clean(credits_data)

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

"""
Now that we know what our data does, let us try to predict tmdb scores based on our final and merged data.
If that doesn't work, let us try to predict it on the cleaned titles_data. We will use Supervised Machine Learning
"""

# Merged Dataset Cleaning

print(modelling_titles_data.info())
print("Null values in columns:\n", modelling_titles_data.isnull().sum(),
      '\n')  # Display number of null values in columns
print("Dimensions: ", modelling_titles_data.shape, '\n')  # Data Structure

print(modelling_titles_data.columns)

# Data Split

y = modelling_titles_data.tmdb_score
features = ['type', 'release_year', 'runtime', 'genres',
            'production_countries', 'imdb_score', 'tmdb_popularity']
X = modelling_titles_data[features]

# Define a function for data category formatting. Select and find distinct values. Convert result into list. Traverse
# that list and change all unique entries into float/int

# X['type'] = X['type'].astype('category')
# X['type'] = X['type'].astype('float64')
# X['genres'] = X['genres'].astype('category')
# X['genres'] = X['genres'].astype('float64')
# X['production_countries'] = X['production_countries'].astype('category')
# X['production_countries'] = X['production_countries'].astype('float64')

# print(X.info())
# print(X['type'])

"""
# Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

# Trial 1 - Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tmdb_model = DecisionTreeRegressor(random_state=1)
tmdb_model.fit(X, y)
predictions = tmdb_model.predict(X)
print(predictions)


# Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
print("Root Mean Squared Error(RMSE) values: ", np.round(rmse, 2))


#calc average first then compare to mea to determine randomstate

# Cross Validation Score
from sklearn.model_selection import \
    cross_val_score  # Choose correct Root Mean Square Deviation by automating an average calculation
scores = cross_val_score(tmdb_model, X, y, scoring='neg_mean_squared_error', cv=5, n_jobs=1)

rmse = np.sqrt(-scores)
print("Root Mean Squared Error(RMSE) values: ", np.round(rmse, 2))
print("Average RMSE: ", np.mean(rmse))
"""
