import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/housing-price-prediction-data/housing_price_dataset.csv")

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['Neighborhood'])],
    remainder='passthrough'
)

sns.pairplot(df, hue='Neighborhood')
plt.show()

numerical_df = df.drop('Neighborhood', axis=1)
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor())
]

for name, model in models:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X = df.drop('Price', axis=1)
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = -cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

    print(f"Model: {name}")
    print(f"Cross-validation Mean MSE: {cv_results.mean()}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse}")
    print(f"Test R-squared: {r2}")

    plt.scatter(y_test, y_test, color='blue', label='Actual')
    plt.scatter(y_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f"{name} - Actual vs Predicted Prices")
    plt.legend()
    plt.show()
