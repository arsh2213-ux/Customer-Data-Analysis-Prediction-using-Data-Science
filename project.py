import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# LOAD DATA

df = pd.read_csv(r"C:/Users/arshk/Downloads/Project/Mall_Customers.csv")

print("Initial Data:\n", df.head())


# DATA CLEANING


# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Drop missing values (if any)
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

print("\nAfter Cleaning:\n", df.shape)


# Using IQR


def remove_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply on numeric columns
for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    df = remove_outliers(col)

print("\nAfter Outlier Removal:\n", df.shape)

# EDA

# 1. Age Distribution
plt.figure()
plt.hist(df['Age'])
plt.title("Age Distribution")
plt.show()

# 2. Income Distribution
plt.figure()
plt.hist(df['Annual Income (k$)'])
plt.title("Income Distribution")
plt.show()

# 3. Spending Score Distribution
plt.figure()
plt.hist(df['Spending Score (1-100)'])
plt.title("Spending Score Distribution")
plt.show()

# 4. Gender Count
plt.figure()
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

# 5. Income vs Spending
plt.figure()
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title("Income vs Spending Score")
plt.show()

# 6. Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# PREPROCESSING (ENCODING)


# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})


# LINEAR REGRESSION


# Features and Target
X = df[['Age', 'Annual Income (k$)', 'Gender']]
y = df['Spending Score (1-100)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n--- MODEL PERFORMANCE ---")
print("R2 Score:", r2)
print("MSE:", mse)

# VISUALIZATION (ACTUAL VS PREDICTED)

plt.figure()
plt.scatter(y_test, y_pred)
plt.title(f"Actual vs Predicted (R2 = {r2:.2f})")
plt.xlabel("Actual Spending Score")
plt.ylabel("Predicted Spending Score")
plt.show()
