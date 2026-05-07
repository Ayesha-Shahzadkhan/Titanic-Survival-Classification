import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("train.csv")

# Basic Info
missing_values = df.isnull().sum()
print("Missing Values: ", missing_values)

print(df.columns)
df = df.drop('Cabin', axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop_duplicates()
print(df['Survived'].value_counts())
missing_values = df.isnull().sum()
print("Missing Values: ", missing_values)

# Encoding categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Feature selection
x = df[['Age','Sex','Pclass','Fare']]
y = df['Survived']

# Logistic Regression
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(x,y)

with open ("model.pkl", "wb") as f:
    pickle.dump(model,f)

with open ("label.pkl", "wb") as f:
    pickle.dump(le,f)