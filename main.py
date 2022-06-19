"""
Phase 1: Exploratory Data Analysis and Data Visualization

This data is information about the shows and movies streaming in the USA.
The intention here is to find out how Netflix content is being consumed by the audience.

Phase 2:

The data has now been cleaned, analyzed and seems usable to try to replicate results with.
The intention is to prepare a Supervised ML model to try to predict the tmdb scores based on given information.
"""

import numpy as np  # To perform linear algebra
import pandas as pd  # To perform data and file processing
import matplotlib.pyplot as plt  # For plotting data
import seaborn as sns  # For visualizing data
from sklearn.model_selection import train_test_split, cross_val_score  # Train Test Split and Average RMSE # calculation
from sklearn.tree import DecisionTreeRegressor  # Decision Tree Algorithm


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

col = ['type', 'genres', 'production_countries']  # Columns that need to be Factorized

# Factorization using pandas
X['type_fac'] = pd.factorize(X['type'])[0]
X['genres_fac'] = pd.factorize(X['genres'])[0]
X['production_countries_fac'] = pd.factorize(X['production_countries'])[0]

factorized_features = ['type_fac', 'release_year', 'runtime', 'genres_fac',
                       'production_countries_fac', 'imdb_score']

X = X[factorized_features]

X.info()  # X is now ready to use

#Trial 1 - Decision Tree Regression
# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

# Trial 1 - Decision Tree Regression

tmdb_model = DecisionTreeRegressor()
tmdb_model.fit(X_train, y_train)
y_pred = tmdb_model.predict(X_test)
print('\n', y_pred)


"""
rmse_track = []
for step in range(100): # looped 100 random states to find least rmse for usability
#We found random_state 10 and 68 to be giving minimum RMSE of 1.08

# Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
rmse = np.round(rmse, 2)
print("Root Mean Squared Error(RMSE) values: ", rmse)
rmse_track.append(rmse)

min_rmse = min(rmse_track)
for i in range(len(rmse_track)):
    if rmse_track[i-1] == min_rmse:
        min_rmse_random_state = i-1
        print("\nMinimum RMSE values in random_state: ", min_rmse_random_state)
        print("Root Mean Squared Error(RMSE) values: ", min_rmse)


# calc average first then compare to mea to determine randomstate

# Cross Validation Score
scores = cross_val_score(tmdb_model, X, y, scoring='neg_mean_squared_error', cv=200, n_jobs=-1)

rmse = np.sqrt(-scores)
print("\nRoot Mean Squared Error(RMSE) values: ", np.round(rmse, 2))
print("\nAverage RMSE: ", np.mean(rmse))

min_rmse = min(rmse)
for i in range(len(rmse)):
    if rmse[i-1] == min_rmse:
        min_rmse_random_state = i-1
        print("\nMinimum RMSE values in random_state: ", min_rmse_random_state)
        print("Root Mean Squared Error(RMSE) values: ", min_rmse)
"""

# Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse = np.round(rmse, 2)
print("Root Mean Squared Error(RMSE) values: ", rmse)


output = pd.DataFrame({'y_test': y_test, 'y_pred': np.round(y_pred, 1)})
output.to_csv('Decision Tree Output Comparison.csv', index=False)
print("Your file was successfully saved!")

#Trial 2 - Random Forest