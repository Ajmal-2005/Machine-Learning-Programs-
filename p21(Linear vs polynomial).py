import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])  # Quadratic relationship

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Plotting
plt.scatter(X, y, color='red', label='Actual')
plt.plot(X, y_lin_pred, color='blue', label='Linear')
plt.plot(X, y_poly_pred, color='green', label='Polynomial (deg=2)')
plt.title("Linear vs Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Comparison
print("Linear Regression MSE:", mean_squared_error(y, y_lin_pred))
print("Polynomial Regression MSE:", mean_squared_error(y, y_poly_pred))
