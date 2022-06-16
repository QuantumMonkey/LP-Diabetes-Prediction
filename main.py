"""
Phase 1: Data Visualization

This data is information about the shows and movies streaming in the USA.
The intention here is to find out how Netflix content is being consumed by the audience.
"""

import numpy as np  # to perform linear algebra
import pandas as pd  # to perform data and file processing
import matplotlib.pyplot as plt # for plotting data
import seaborn as sns # for visualizing data

# Import files into DataFrames
titles_data = pd.read_csv('Dataset/titles.csv')
credits_data = pd.read_csv('Dataset/credits.csv')
