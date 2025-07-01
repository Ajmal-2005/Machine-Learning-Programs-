import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example: Monthly sales data
data = {'Month': [1, 2, 3, 4, 5, 6], 'Sales': [250, 270, 300, 310, 400, 420]}
df = pd.DataFrame(data)

# Features and Target
X = df[['Month']]
y = df['Sales']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict sales for next 2 months
future_months = pd.DataFrame({'Month': [7, 8]})
future_sales = model.predict(future_months)
print("Future Sales:", future_sales)

# Optional plot
plt.plot(df['Month'], y, label='Past Sales', marker='o')
plt.plot(future_months['Month'], future_sales, label='Predicted Sales', marker='o', color='red')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales Forecast')
plt.legend()
plt.grid(True)
plt.show()
