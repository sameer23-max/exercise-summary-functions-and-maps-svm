import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
try:
    df = pd.read_csv('/kaggle/input/datasets/sameerml777/drug-csv/drug200.csv')
except FileNotFoundError:
    df = pd.read_csv('drug200.csv')

print(df.head())
print('\nMissing values:')
print(df.isnull().sum())

# Encode categorical columns
# Sex: M=1, F=0
# BP: LOW=0, NORMAL=1, HIGH=2
# Cholesterol: NORMAL=0, HIGH=1
# Drug: encode labels with LabelEncoder

df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['BP'] = df['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
df['Cholesterol'] = df['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})

le = LabelEncoder()
df['Drug'] = le.fit_transform(df['Drug'])

print('\nEncoded data preview:')
print(df.head())

# Split features and target
X = df.drop(columns='Drug')
y = df['Drug']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale data before model training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel='rbf', C=1)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'\nModel accuracy: {accuracy:.4f}')

# Optional: view files in Kaggle input dir
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
