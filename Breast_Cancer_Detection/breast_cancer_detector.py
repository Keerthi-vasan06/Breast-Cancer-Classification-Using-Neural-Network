import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle# Load the dataset
df = pd.read_csv('data.csv')
# Display the first 10 rows
print(df.head(10))
# General info about the dataset
print(df.info())
# Check for missing values
print(df.isna().sum())
# Statistical summary of the dataset
print(df.describe())
# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)
# Display the first 10 rows after dropping columns
print(df.head(10))
# Shape of the dataset
print(df.shape)
# Count of each diagnosis type
print(df['diagnosis'].value_counts())
# Plot diagnosis counts
sns.countplot(df['diagnosis'], label="count")
plt.show()
# Encode 'diagnosis' column to numeric (B -> 0, M -> 1)
lb = LabelEncoder()
df['diagnosis'] = lb.fit_transform(df['diagnosis'])
# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
# Pairplot visualization
sns.pairplot(df.iloc[:, :5], hue="diagnosis")
plt.show()
# Split data into features and target
X = df.iloc[:, 1:].values  # Features (all except diagnosis)
y = df['diagnosis'].values  # Target (diagnosis)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use transform only on test data
# Logistic Regression model
log = LogisticRegression()
log.fit(X_train, y_train)
# Model evaluation
train_score = log.score(X_train, y_train)
print(f"Training Accuracy: {train_score}")
y_pred = log.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy}")
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(log, f)