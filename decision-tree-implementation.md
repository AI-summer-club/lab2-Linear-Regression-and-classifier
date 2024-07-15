# Decision Tree Classifier Implementation

In this section, we'll walk through the process of implementing a Decision Tree classifier using Python and the scikit-learn library. We'll use the Iris dataset, which is a classic dataset for classification tasks.

## 1. Data Preparation

### Loading the Data

First, we need to load our dataset:

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Add species names
y = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
```

### Exploring the Data

Let's take a look at our data:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Display basic information
print(X.info())
print(X.describe())

# Visualize the distribution of features
plt.figure(figsize=(12, 10))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=X, x=feature, hue=y, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(pd.concat([X, y], axis=1), hue='species')
plt.show()
```

## 2. Data Preprocessing

For this dataset, we don't need to do much preprocessing as it's already clean and normalized. However, in real-world scenarios, you might need to handle missing values, encode categorical variables, and scale features.

### Splitting the Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Model Training

```python
from sklearn.tree import DecisionTreeClassifier

# Create and train the model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
```

## 4. Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## 5. Visualizing the Decision Tree

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(dt_classifier, feature_names=iris.feature_names, class_names=iris.target