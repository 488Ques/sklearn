from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
california = fetch_california_housing()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=42)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the test set and evaluate the model
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse) # Mean Squared Error: 0.5558915986952429
