import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

# Load the dataset
df = pd.read_csv('HumanResources.csv', sep=';')

# Step 1: Data Preprocessing
df['attrition'] = df['Termdate'].notnull().astype(int)
df['Hiredate'] = pd.to_datetime(df['Hiredate'])
df['Termdate'] = pd.to_datetime(df['Termdate'], errors='coerce')
df['Birthdate'] = pd.to_datetime(df['Birthdate'])

# Step 2: Feature Engineering
df['age'] = (pd.Timestamp('today') - df['Birthdate']).dt.days // 365
df['tenure'] = (df['Termdate'].fillna(pd.Timestamp('today')) - df['Hiredate']).dt.days / 365

performance_map = {
    'Excellent': 4,
    'Good': 3,
    'Satisfactory': 2,
    'Needs Improvement': 1
}
df['Performance Rating'] = df['Performance Rating'].map(performance_map)

# Step 3: Select relevant features
features = ['Gender', 'Education Level', 'age', 'tenure', 'Department', 'Job Title', 'Salary', 'Performance Rating']
df = pd.get_dummies(df, columns=['Gender', 'State', 'City', 'Education Level', 'Department', 'Job Title', 'Performance Rating'], drop_first=True)

# Step 4: Define target and features
X = df.drop(columns=['Employee_ID', 'First Name', 'Last Name', 'Birthdate', 'Hiredate', 'Termdate', 'attrition'])
y = df['attrition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Apply Random Under Sampling to the training data
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_smote, y_train_smote)

# Step 5: Train the XGBClassifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_rus, y_train_rus)

# Step 6: Make Predictions
y_pred = xgb_model.predict(X_test)

# Step 7: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
importances = xgb_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(importance_df.head())
