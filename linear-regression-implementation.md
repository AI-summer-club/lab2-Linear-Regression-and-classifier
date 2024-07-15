# Linear Regression Implementation

In this section, we'll walk through the process of implementing a linear regression model using Python and the scikit-learn library. We'll use the House Prices dataset from Kaggle.

## 1. Data Preparation

### Loading the Data

First, we need to load our dataset:

```python
import pandas as pd

# Load the data
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
```

### Exploring the Data

It's crucial to understand our data before we start modeling:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Display basic information
print(train_df.info())

# Summary statistics
print(train_df.describe())

# Visualize the target variable (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Sale Price Distribution')
plt.show()

# Correlation analysis
numeric_columns = train_df.select_dtypes(include=['int64', 'float64'])
correlations = numeric_columns.corr()['SalePrice'].sort_values(ascending=False)
print(correlations.head(10))  # Top 10 positive correlations
print(correlations.tail(10))  # Top 10 negative correlations

# Visualize relationship between a feature and the target
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='GrLivArea', y='SalePrice')
plt.title('Relationship between Ground Living Area and Sale Price')
plt.show()
```

## 2. Data Preprocessing

### Handling Missing Values and Encoding

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Separate numeric and categorical features
numeric_columns = train_df.select_dtypes(include=['int64', 'float64'])
categorical_columns = train_df.select_dtypes(include=['object'])

# Handle missing values in numeric columns
imputer = SimpleImputer(strategy='mean')
numeric_columns_imputed = pd.DataFrame(imputer.fit_transform(numeric_columns), columns=numeric_columns.columns)

# One-Hot Encoding for categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_categorical = encoder.fit_transform(categorical_columns)
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns.columns))

# Combine the encoded categorical features with the imputed numeric features
X = pd.concat([numeric_columns_imputed, encoded_df], axis=1)
```

### Feature Scaling

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

### Splitting the Data

```python
from sklearn.model_selection import train_test_split

y = train_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

## 3. Model Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

## 4. Model Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual House Prices')
plt.show()
```

## 5. Interpreting the Model

```python
# Get feature importances
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(10))

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Most Important Features')
plt.show()
```

This implementation guide provides a step-by-step approach to building a linear regression model, from data preparation to model interpretation. It's important to emphasize that this is an iterative process, and you may need to go back and forth between these steps to improve your model.
