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
Now that we know what our data does, let us try to predict tmdb scores based on our final and merged data.
If that doesn't work, let us try to predict it on the cleaned titles_data. We will use Supervised Machine Learning.
"""

# Factorization using pandas
modelling_titles_data['type_fac'] = pd.factorize(modelling_titles_data['type'])[0]
modelling_titles_data['genres_fac'] = pd.factorize(modelling_titles_data['genres'])[0]
modelling_titles_data['production_countries_fac'] = pd.factorize(modelling_titles_data['production_countries'])[0]
# Data Split

y = modelling_titles_data.tmdb_score
features = ['type_fac', 'release_year', 'runtime', 'genres_fac',
            'production_countries_fac', 'imdb_score']
X = modelling_titles_data[features]

X.info()  # X is now ready to use

# Gaussian plot


# Trial 1 - Decision Tree Regression
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

# Trial 2 - Random Forest
