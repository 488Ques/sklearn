from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dữ liệu từ Scikit-learn
california = fetch_california_housing()

# Chia thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=42)

# Huấn luyện mô hình linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Đánh giá mô hình bằng mean squared error, sử dụng tập kiểm tra
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse) # Mean Squared Error: 0.5558915986952429
