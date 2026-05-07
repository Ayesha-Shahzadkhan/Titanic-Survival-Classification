import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

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

# Train-Test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Logistic Regression
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(x_train,y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracuy
print("Accuracy: ", accuracy_score(y_test,y_pred))
print("Classification Report: ", classification_report(y_test,y_pred))
 
scores = cross_val_score(model, x, y, cv=5)
print("Cross-Val Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
