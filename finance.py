# Generated from: finance (1).ipynb
# Converted at: 2026-04-29T03:45:58.952Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
path = '/kaggle/input/datasets/jumpingdino/german-credit-dataset/german_credit_data.csv'
df = pd.read_csv(path)

print("Dataset shape:", df.shape)
df.head()

# 3. Basic Info
print(df.info())
print(df.describe())

# 4. Missing Values
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns
# 5. EDA

# Target distribution
sns.countplot(x='target', data=df)
plt.title('Target Distribution')
plt.show()

# Age distribution
plt.hist(df['age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Credit amount
plt.hist(df['credit_amount'], bins=30)
plt.title('Credit Amount Distribution')
plt.show()

# Duration vs Target
sns.boxplot(x='target', y='month_duration', data=df)
plt.title('Duration vs Target')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 6. Preprocessing

# Encode target
df['target'] = df['target'].map({'good':1, 'bad':0})

# One-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Split data
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# 9. Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Top 10 Important Features')
plt.show()

# 10. Insights
print("""
INSIGHTS:
1. Credit amount and duration strongly affect credit risk.
2. Customers with poor credit history are more likely to be classified as 'bad'.
3. Higher age groups tend to be safer borrowers.
4. Employment duration shows a positive correlation with creditworthiness.
5. Logistic Regression outperforms Random Forest in this dataset, suggesting that the relationships are relatively linear.
6. Both models struggle to accurately detect high-risk customers (class 0), indicating a class imbalance issue.
""")

# 11. Conclusion
print("""
CONCLUSION:
This project develops a credit risk prediction model using machine learning techniques.

Logistic Regression achieved better performance (80.5% accuracy) compared to Random Forest (79%), indicating that simpler linear models can be effective for this dataset.

However, both models show limitations in identifying high-risk customers, highlighting the need for handling class imbalance and improving recall for the minority class.

Future work includes applying resampling techniques (SMOTE) and experimenting with advanced models such as XGBoost to enhance performance.
""")