import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import skew

# Step 1: Read the dataset
df = pd.read_csv("cars.csv")

# Display the DataFrame before filling missing values
print("DataFrame before filling missing values:")
print(df.head())

# Display the number of features and examples
print(f"Number of features: {len(df.columns)}")
print(f"Number of examples: {len(df)}")

# Step 2: Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per feature:")
print(missing_values)

# Step 3: Drop non-numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Display the mean values for each numeric feature
print("\nMean values for each numeric feature:")
print(numeric_df.mean())

# Display the mode values for each non-numeric feature
non_numeric_df = df.select_dtypes(exclude=[np.number])
print("\nMode values for each non-numeric feature:")
print(non_numeric_df.mode().iloc[0])  # Assuming there is a unique mode, use iloc[0]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

# Display the DataFrame after filling missing values
print("\nDataFrame after filling missing values:")
print(df_imputed.head())

# Display summary statistics for each feature
print("\nSummary statistics for each feature:")
print(df_imputed.describe())

# Step 4: Box plot for fuel economy by country
plt.figure(figsize=(10, 6))
sns.boxplot(x="origin", y="mpg", data=df)
plt.title("Fuel Economy by Country")
plt.show()

# Step 5: Histograms for 'acceleration', 'horsepower', and 'mpg' with skewness
plt.figure(figsize=(12, 8))

# Acceleration Histogram
plt.subplot(3, 2, 1)
plt.hist(df_imputed['acceleration'], bins=20, edgecolor='black')
plt.title('Acceleration Histogram')

# Horsepower Histogram
plt.subplot(3, 2, 2)
plt.hist(df_imputed['horsepower'], bins=20, edgecolor='black')
plt.title('Horsepower Histogram')

# MPG Histogram
plt.subplot(3, 2, 3)
plt.hist(df_imputed['mpg'], bins=20, edgecolor='black')
plt.title('MPG Histogram')

# Perform skewness test for pairs of features
def skewness_test(feature1, feature2):
    skewness = skew(feature1 - feature2)
    return skewness

plt.tight_layout()
plt.show()

# Step 6: Quantitative measure for distribution similarity (e.g., skewness)
skewness_1 = skewness_test(df_imputed['acceleration'], df_imputed['horsepower'])
skewness_2 = skewness_test(df_imputed['acceleration'], df_imputed['mpg'])
skewness_3 = skewness_test(df_imputed['horsepower'], df_imputed['mpg'])

print(f'\nSkewness: Acceleration vs. Horsepower - Skewness: {skewness_1}')
print(f'Skewness: Acceleration vs. MPG - Skewness: {skewness_2}')
print(f'Skewness: Horsepower vs. MPG - Skewness: {skewness_3}\n')

# Step 7: Scatter plot for 'horsepower' vs 'mpg'
plt.figure(figsize=(10, 6))
plt.scatter(df_imputed['horsepower'], df_imputed['mpg'])
plt.title('Scatter plot of Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()

# Step 8: Closed-form linear regression
X = df_imputed[['horsepower']]
y = df_imputed['mpg']

# Standardize the features
scaler_linear = StandardScaler()
X_standardized = scaler_linear.fit_transform(X)

# Add a column of ones for the intercept term
X_linear = np.c_[np.ones(X_standardized.shape[0]), X_standardized]

# Compute the closed-form solution
theta_linear = np.linalg.inv(X_linear.T @ X_linear) @ X_linear.T @ y

# Print the values of w0 and w1
print(f'Intercept (w0): {theta_linear[0]}')
print(f'Coefficient for Horsepower (w1): {theta_linear[1]}\n')

# Plotting the learned linear line
plt.figure(figsize=(10, 6))
plt.scatter(df_imputed['horsepower'], df_imputed['mpg'])
plt.title('Linear Regression: Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.plot(df_imputed['horsepower'], X_linear @ theta_linear, color='red', linewidth=2)
plt.show()

# Suppress PolynomialFeatures warning
warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)

# Step 9: Quadratic function regression
# Adding a quadratic term for 'horsepower'
X_quad = df_imputed[['horsepower']]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_quad = poly.fit_transform(X_quad)
columns = [f'horsepower_{i}' for i in range(X_quad.shape[1])]
X_quad = pd.DataFrame(X_quad, columns=columns)

# Fit the quadratic model
model_quad = LinearRegression()
model_quad.fit(X_quad, y)

# Print the values of w0, w1, and w2
print(f'Intercept (w0): {model_quad.intercept_}')
print(f'Coefficient for Horsepower (w1): {model_quad.coef_[0]}')
print(f'Coefficient for Horsepower^2 (w2): {model_quad.coef_[1]}\n')

# Plotting the learned quadratic curve
plt.figure(figsize=(10, 6))
plt.scatter(df_imputed['horsepower'], df_imputed['mpg'])
plt.title('Quadratic Regression: Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')

# Generate points on the curve for plotting
x_curve = np.linspace(df_imputed['horsepower'].min(), df_imputed['horsepower'].max(), 100)
x_curve_quad = poly.transform(x_curve.reshape(-1, 1))
y_curve_quad = model_quad.predict(x_curve_quad)

plt.plot(x_curve, y_curve_quad, color='red', linewidth=2)
plt.show()

# Step 10: Gradient Descent for Simple Linear Regression
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        theta = theta - (1/m) * learning_rate * (X.T @ errors)

    return theta

# Setting up the data for gradient descent (simple linear regression)
X_grad = df_imputed[['horsepower']]
X_grad = np.c_[np.ones(X_grad.shape[0]), X_grad]
y_grad = df_imputed['mpg']
theta_grad = np.zeros(X_grad.shape[1])

# Hyperparameters for gradient descent
learning_rate = 0.01
iterations = 500

# Running gradient descent after standardizing features
scaler = StandardScaler()
X_grad_scaled = scaler.fit_transform(X_grad[:, 1].reshape(-1, 1))  # Standardize only the feature, excluding the intercept
X_grad_scaled = np.c_[np.ones(X_grad_scaled.shape[0]), X_grad_scaled]
theta_grad = np.zeros(X_grad_scaled.shape[1])

theta_grad = gradient_descent(X_grad_scaled, y_grad, theta_grad, learning_rate, iterations)

# Print the values of w0 and w1
print(f'Intercept (w0) after gradient descent: {theta_grad[0]}')
print(f'Coefficient for Horsepower (w1) after gradient descent: {theta_grad[1]}')

# Plotting the learned line using gradient descent
plt.figure(figsize=(10, 6))
plt.scatter(df_imputed['horsepower'], df_imputed['mpg'])
plt.title('Gradient Descent (Simple Linear Regression): Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')

# Generate points on the line for plotting
x_line_grad = np.linspace(df_imputed['horsepower'].min(), df_imputed['horsepower'].max(), 100)
x_line_grad_scaled = scaler.transform(x_line_grad.reshape(-1, 1))  # Scale the test data
x_line_grad_scaled = np.c_[np.ones(x_line_grad_scaled.shape[0]), x_line_grad_scaled]
y_line_grad = x_line_grad_scaled @ theta_grad

plt.plot(x_line_grad, y_line_grad, color='red', linewidth=2)
plt.show()

# Calculate and print the correlation coefficient
correlation_coefficient = np.corrcoef(df_imputed['horsepower'], df_imputed['mpg'])[0, 1]
print(f'\nCorrelation Coefficient between Horsepower and MPG: {correlation_coefficient}')

# Print whether the correlation is positive or negative
if correlation_coefficient > 0:
    print('The correlation is positive.')
elif correlation_coefficient < 0:
    print('The correlation is negative.')
else:
    print('There is no correlation.')
