import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle  # Import pickle for model saving

# Load data (replace with your file path)
df = pd.read_csv('generated_NDP_data-2.csv')

# Sample 2000 rows randomly
sampled_df = df.sample(n=2000, replace=True)

# Data Preprocessing
# Removing null values
sampled_df.dropna(inplace=True)

# EDA
# Pie chart for Disaster Type
sampled_df['DisasterType'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

# Histogram for Temperature (°C)
plt.hist(sampled_df['Temperature'], bins=20)
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

# Bar graph for Seismic Activity (Richter scale)
sampled_df['SeismicActivity'].value_counts().plot.bar()
plt.xlabel('SeismicActivity')
plt.ylabel('Count')
plt.show()

# Scatter plot for Temperature (°C) vs Humidity (%)
plt.scatter(sampled_df['Temperature'], sampled_df['Humidity'])
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()

# Feature Selection
features = sampled_df[['Latitude', 'Longitude', 'Temperature', 'Humidity', 
                       'Pressure', 'WindSpeed', 'Precipitation', 
                       'SeismicActivity']]
target = sampled_df['DisasterType']

# Encoding 'Disaster Type' column
le = LabelEncoder()
target_encoded = le.fit_transform(target)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Model Training
# RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Hyperparameter Tuning for RandomForest
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 4]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# Evaluate RandomForest
rf_accuracy, rf_report = evaluate_model(rf_model, X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy}\n{rf_report}")

# Evaluate DecisionTree
dt_accuracy, dt_report = evaluate_model(dt_model, X_test, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy}\n{dt_report}")

# Compare and select best model based on evaluation metrics
# You can add more models and comparisons as needed
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
