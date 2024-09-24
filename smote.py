import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

# Load the dataset
df = pd.read_csv('HumanResources.csv', sep=';')

# Step 1: Data Preprocessing
# Create an 'attrition' column: if Termdate is not null, the employee left (attrition = 1), else they stayed (attrition = 0)
df['attrition'] = df['Termdate'].notnull().astype(int)

# Convert dates to datetime
df['Hiredate'] = pd.to_datetime(df['Hiredate'])
df['Termdate'] = pd.to_datetime(df['Termdate'], errors='coerce')  # Handle missing termination dates
df['Birthdate'] = pd.to_datetime(df['Birthdate'])

# Step 2: Feature Engineering
# Calculate age from Birthdate
df['age'] = (pd.Timestamp('today') - df['Birthdate']).dt.days // 365

# Calculate tenure as the difference between hire date and termination date (or current date if still employed)
df['tenure'] = (df['Termdate'].fillna(pd.Timestamp('today')) - df['Hiredate']).dt.days / 365  # tenure in years

# Map performance rating to numerical values
performance_map = {
    'Excellent': 4,
    'Good': 3,
    'Satisfactory': 2,
    'Needs Improvement': 1
}
df['Performance Rating'] = df['Performance Rating'].map(performance_map)

# Step 3: Select relevant features
features = ['Gender', 'Education Level', 'age', 'tenure', 'Department', 'Job Title', 'Salary', 'Performance Rating']

# Convert categorical variables to numerical
# Apply get_dummies to relevant categorical columns
df = pd.get_dummies(df, columns=['Gender', 'State', 'City', 'Education Level', 'Department', 'Job Title', 'Performance Rating'], drop_first=True)

df['Hiredate'] = pd.to_datetime(df['Hiredate'], dayfirst=True)
df['Birthdate'] = pd.to_datetime(df['Birthdate'], dayfirst=True)
df['Termdate'] = pd.to_datetime(df['Termdate'], dayfirst=True, errors='coerce')

# Calculate age
df['Age'] = (pd.to_datetime('today') - df['Birthdate']).dt.days // 365

df['attrition'] = df['Termdate'].notnull().astype(int)

# Step 4: Define target and features
X = df.drop(columns=['Employee_ID', 'First Name', 'Last Name', 'Birthdate', 'Hiredate', 'Termdate', 'attrition'])
y = df['attrition']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Apply Random Under Sampling to the training data
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_smote, y_train_smote)

# Step 5: Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rus, y_train_rus)

# Step 6: Make Predictions
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Feature Importance
importances = rf_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(importance_df.head())
