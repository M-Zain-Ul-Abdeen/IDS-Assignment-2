import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv")
df.head(20)
df.describe()
df.isnull().sum()
df.dropna(inplace=True)

plt.figure(figsize=(10, 6))
sns.histplot(df['Average User Rating'], bins=20, kde=True)
plt.title('Distribution of Average User Ratings')
plt.xlabel('Average User Rating')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average User Rating', y='Price')
plt.title('Average User Rating vs Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average User Rating', y='User Rating Count')
plt.title('Average User Rating vs User Rating Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(y='Primary Genre', data=df, order=df['Primary Genre'].value_counts().index)
plt.title('Distribution of App Genres')
plt.xlabel('Count')
plt.ylabel('Primary Genre')
plt.show()

df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)

numerical_cols = ['Average User Rating', 'User Rating Count', 'Price']
cleaned_df = df.copy()
for col in numerical_cols:
    Q1 = cleaned_df[col].quantile(0.25)
    Q3 = cleaned_df[col].quantile(0.75)
    IQR = Q3 - Q1
    cleaned_df = cleaned_df[~((cleaned_df[col] < (Q1 - 1.5 * IQR)) | (cleaned_df[col] > (Q3 + 1.5 * IQR)))]

plt.figure(figsize=(10, 6))
sns.histplot(cleaned_df['Average User Rating'], bins=20, kde=True, color='blue', stat='density')
mu, sigma = norm.fit(cleaned_df['Average User Rating'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Distribution of Average User Ratings (After Cleaning)')
plt.xlabel('Average User Rating')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=cleaned_df, x='Average User Rating', y='Price')
plt.title('Average User Rating vs Price (After Cleaning)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=cleaned_df, x='Average User Rating', y='User Rating Count')
plt.title('Average User Rating vs User Rating Count (After Cleaning)')
plt.show()

numerical_corr = cleaned_df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(numerical_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Numerical Columns')
plt.show()

X = df[['User Rating Count', 'Price']]
y = df['Average User Rating']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_regressor, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

rf_regressor.fit(X_train, y_train)

y_pred_train = rf_regressor.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

y_pred_test = rf_regressor.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Cross-validation RMSE scores:", cv_rmse_scores)
print("Mean CV RMSE:", np.mean(cv_rmse_scores))

feature_importances = rf_regressor.feature_importances_
print("Feature Importances:", feature_importances)
