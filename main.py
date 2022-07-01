"""
Diabetes Prediction using Regression (Supervised Machine Learning)
"""

# Import Packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


def lr_prediction(ind_train, ind_test, target_train):
    """

    :param ind_train:
    :param ind_test:
    :param target_train:
    :return: LR model
    """
    LR_model = LogisticRegression(random_state=0, verbose=True)
    LR_model.fit(ind_train, target_train)
    target_prediction = LR_model.predict(ind_test)
    return target_prediction


def nb_prediction(ind_train, ind_test, target_train):
    """

    :param ind_train:
    :param ind_test:
    :param target_train:
    :return: Gaussian NB model
    """
    NB_model = GaussianNB()
    NB_model.fit(ind_train, target_train)
    target_prediction = NB_model.predict(ind_test)
    return target_prediction


def rf_prediction(ind_train, ind_test, target_train):
    """
    Random Forest Classifier Algorithm
    :param ind_train:
    :param ind_test:
    :param target_train:
    :return: RF model
    """
    RF_model = RandomForestClassifier(n_estimators=300, criterion='entropy',
                                      min_samples_split=10, random_state=0, verbose=True)

    RF_model.fit(ind_train, target_train)
    target_prediction = RF_model.predict(ind_test)
    return target_prediction


def plot_confusionmatrix(target_test, target_prediction):
    cm = confusion_matrix(target_test, target_prediction)
    display = ConfusionMatrixDisplay(cm)
    display.plot(cmap="Reds")
    plt.show()


def class_report(target_test, target_prediction):
    print(classification_report(target_test, target_prediction))


# Data Source
df = pd.read_csv(r'diabetes_health_indicators.csv')

# Display dataset information
df.info()

df.corr()

plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.show()

# Check for null values
sns.heatmap(df.isnull(), cmap="Blues")

df.isnull().sum()

# EDA
# Display all columns in histogram format
df.hist(figsize=(20, 20))

# Checking distribution of target variable
labels = 'Healthy', 'Diabetic', 'Pre-Diabetic'
ex = [0.1, 0.1, 0.1]
df.Diabetes_012.value_counts().plot.pie(labels=labels, autopct='%1.2f%%', shadow=True, explode=ex)
plt.show()

# Checking duplicates
duplicates = df[df.duplicated()]
print("Duplicates: ", len(duplicates))

# Drop duplicates
df.drop_duplicates(inplace=True)

# Removing PhysHlth as it has a high correlation similar to GenHlth.
df.drop('PhysHlth', inplace=True, axis=1)

# Counts of the target variable
sns.countplot(data=df, y='Diabetes_012')
plt.show()

# Checking number of records (samples)
non_diabetic = df[df['Diabetes_012'] == 0]
pre_diabetic = df[df['Diabetes_012'] == 1]
diabetic = df[df['Diabetes_012'] == 2]

# Cloning samples (oversampling) to balance the dataset to train model correctly
pre_diabetic_os = pre_diabetic.sample(len(non_diabetic), replace=True)
diabetic_os = diabetic.sample(len(non_diabetic), replace=True)

df_new = pd.concat([pre_diabetic_os, diabetic_os, non_diabetic], axis=0)
df_new['Diabetes_012'].value_counts()

# Counts of the oversampled target variable
sns.countplot(data=df_new, y='Diabetes_012')
plt.show()

# Train-test split

# input
x = df_new.iloc[:, 1:].values
# output
y = df_new.iloc[:, 0].values

# Split data into 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling using MinMaxScalar
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Principal Component Analysis
pca = PCA(n_components=20)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

# Logistic Regression Algorithm
y_pred = lr_prediction(x_train, x_test, y_train)
plot_confusionmatrix(y_test, y_pred)
print("Accuracy of Logistic Regression model is ", accuracy_score(y_test, y_pred) * 100, "%.")
class_report(y_test, y_pred)

# Naive-Bayes Classifier Algorithm
y_pred = nb_prediction(x_train, x_test, y_train)
plot_confusionmatrix(y_test, y_pred)
print("Accuracy of Gaussian Naive Bayes model is ", accuracy_score(y_test, y_pred) * 100, "%.")
class_report(y_test, y_pred)

# Random Forest Classifier Algorithm
y_pred = rf_prediction(x_train, x_test, y_train)
plot_confusionmatrix(y_test, y_pred)
print("Accuracy of Random Forest model is ", accuracy_score(y_test, y_pred) * 100, "%.")
class_report(y_test, y_pred)
