# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pK7w8rBdJ7ztaBUs__cibaKMIFWFB6m6
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create a DataFrame with mixed data types
data = {'Category': ['A', 'B', 'A', 'C'],
        'Age': [25, 35, 45, 55],
        'Salary': [50000, 60000, 70000, 80000]}

df = pd.DataFrame(data)

# Apply StandardScaler on numerical columns
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print(df)

from sklearn.preprocessing import OneHotEncoder

# Initialize encoder with correct parameter name
encoder = OneHotEncoder(sparse_output=False)

# Apply encoding
encoded_data = encoder.fit_transform(df[['Category']])

# Convert back to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Category']))

# Concatenating with original DataFrame (excluding original categorical column)
df = pd.concat([df.drop(columns=['Category']), encoded_df], axis=1)

print(df)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Decision Tree Accuracy: {accuracy:.2f}")

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM Model Accuracy: {accuracy:.2f}")
