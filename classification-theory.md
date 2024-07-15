# Classification Theory

## What is Classification?

Classification is a supervised learning technique where the goal is to predict the categorical class labels of new instances, based on past observations. Unlike regression, which predicts continuous values, classification is used for predicting discrete categories.

## Key Concepts

1. **Binary vs. Multi-class Classification**:
   - Binary: Two possible classes (e.g., spam or not spam)
   - Multi-class: More than two classes (e.g., classifying types of flowers)

2. **Features and Labels**:
   - Features: The input variables used to make predictions
   - Labels: The output categories we're trying to predict

3. **Decision Boundary**: The line or surface that separates different classes in the feature space

4. **Probability Estimation**: Many classifiers can provide probability estimates for each class

5. **Common Classification Algorithms**:
   - Logistic Regression
   - Decision Trees
   - Random Forests
   - Support Vector Machines (SVM)
   - K-Nearest Neighbors (KNN)
   - Neural Networks

## Focus on Decision Trees

In this lecture, we'll focus on Decision Trees as our classification algorithm.

### What is a Decision Tree?

A Decision Tree is a flowchart-like tree structure where:
- Each internal node represents a feature (attribute)
- Each branch represents a decision rule
- Each leaf node represents an outcome (class label)

### How Decision Trees Work:

1. **Tree Construction**: The algorithm starts at the root node and splits the data based on the feature that results in the largest information gain.

2. **Splitting Criteria**: Common methods include:
   - Gini Impurity: Measures the probability of incorrect classification
   - Entropy: Measures the impurity or uncertainty in the data

3. **Pruning**: Reducing the size of the tree to prevent overfitting

4. **Prediction**: To classify a new instance, we traverse the tree from root to leaf, following the path determined by the instance's features.

### Advantages of Decision Trees:

- Easy to understand and interpret
- Requires little data preparation
- Can handle both numerical and categorical data
- Performs well with large datasets

### Disadvantages:

- Can create overly complex trees that do not generalize well (overfitting)
- Can be unstable because small variations in the data might result in a completely different tree
- Biased toward features with more levels (in the case of categorical variables)

## Model Evaluation

Common metrics for evaluating classification models include:

1. **Accuracy**: The proportion of correct predictions among the total number of cases examined

2. **Precision**: The proportion of true positive predictions among all positive predictions

3. **Recall**: The proportion of true positive predictions among all actual positive cases

4. **F1 Score**: The harmonic mean of precision and recall

5. **Confusion Matrix**: A table showing correct and incorrect predictions, broken down by class

6. **ROC Curve and AUC**: Visualizes the performance of a binary classifier system as its discrimination threshold is varied

In our lab, we'll be using a Decision Tree classifier to predict the species of Iris flowers based on their features.
