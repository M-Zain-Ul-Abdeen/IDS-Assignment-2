import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/hr-analytics/HR_comma_sep.csv")
df.columns = ['satisfaction', 'evaluation', 'projectCount', 'averageMonthlyHours', 'yearsAtCompany', 'workAccident', 'left', 'hadPromotion', 'department', 'salary']

df_numeric = df.select_dtypes(exclude='object')
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='Blues')
plt.title('Correlation matrix of employee retention data')
plt.show()

selected_vars = ['satisfaction', 'projectCount', 'yearsAtCompany', 'workAccident', 'hadPromotion']

plt.figure(figsize=(8, 6))
sns.barplot(x='salary', y='left', data=df)
plt.title('Employee retention by salary level')
plt.xlabel('Salary level')
plt.ylabel('Proportion of employees who left')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='department', y='left', data=df)
plt.title('Employee retention by department')
plt.xlabel('Department')
plt.ylabel('Proportion of employees who left')
plt.xticks(rotation=45)
plt.show()

X = df[selected_vars]
y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
