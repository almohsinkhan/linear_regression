# Applying linear regression from scratch using numpy
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X =  np.random.rand(100, 1) * 10  # 100 random points in [0, 10)
y = 3.5 * X + np.random.rand(100, 1) * 5  # Linear relation with noise

# Add bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Linear Regression using Normal Equation

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Estimated coefficients (intercept and slope):", theta_best.ravel())

# linear Regression using Gradient Descent
learning_rate = 0.01
n_iterations = 1000
m = X_b.shape[0]
theta = np.random.randn(2, 1)  # random initialization
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients
print("Estimated coefficients from Gradient Descent (intercept and slope):", theta.ravel())

# Plotting the results
plt.scatter(X, y, color='blue', label='Data points')
X_new = np.array([[0], [10]])   # New data for prediction    
X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]  # add bias term
y_predict = X_new_b.dot(theta_best)  # Predictions using Normal Equation
plt.plot(X_new, y_predict, color='red', label='Regression Line (Normal Equation)')
y_predict_gd = X_new_b.dot(theta)  # Predictions using Gradient Descent
plt.plot(X_new, y_predict_gd, color='green', linestyle='--', label='Regression Line (Gradient Descent)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression from Scratch')     


# make a genarlized linear regression function
def linear_regression(X, y, method='normal', learning_rate=0.01, n_iterations=1000):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
    if method == 'normal':
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    elif method == 'gradient_descent':
        m = X_b.shape[0]
        theta = np.random.randn(2, 1)  # random initialization
        for iteration in range(n_iterations):
            gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
            theta -= learning_rate * gradients
    else:
        raise ValueError("Method must be 'normal' or 'gradient_descent'")
    return theta
