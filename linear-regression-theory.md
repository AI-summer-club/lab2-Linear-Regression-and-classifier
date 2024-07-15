# Linear Regression Theory

## What is Linear Regression?

Linear regression is a fundamental statistical and machine learning technique used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables.

### The Linear Regression Equation

The basic form of a linear regression model is:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + ε
```

Where:
- `y` is the dependent variable (what we're trying to predict)
- `x₁, x₂, ...` are the independent variables (features)
- `β₀` is the y-intercept (the value of y when all x's are 0)
- `β₁, β₂, ...` are the coefficients (weights) for each independent variable
- `ε` is the error term (the difference between the predicted and actual y values)

## Key Concepts

1. **Best Fit Line**: The goal is to find the line that best fits the data points, minimizing the overall error.

2. **Ordinary Least Squares (OLS)**: This is the most common method for estimating the coefficients. It minimizes the sum of the squared differences between the observed and predicted values.

3. **Assumptions**:
   - Linearity: The relationship between X and Y is linear
   - Independence: Observations are independent of each other
   - Homoscedasticity: The variance of residual is the same for any value of X
   - Normality: For any fixed value of X, Y is normally distributed

4. **Types of Linear Regression**:
   - Simple Linear Regression: One independent variable
   - Multiple Linear Regression: Two or more independent variables

5. **Model Evaluation**:
   - R-squared (R²): Measures the proportion of variance in the dependent variable explained by the independent variables
   - Mean Squared Error (MSE): Average squared difference between the estimated values and actual value
   - Root Mean Squared Error (RMSE): Square root of MSE, in the same units as the dependent variable

## Applications

Linear regression is widely used in various fields:
- Economics: Predicting economic trends
- Finance: Stock price prediction
- Real Estate: House price estimation
- Marketing: Sales forecasting
- And many more!

In our lab, we'll be using linear regression to predict house prices based on various features of the houses.
