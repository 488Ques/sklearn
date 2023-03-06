import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

# lấy features của bộ dữ liệu
X = iris.data[:, :2]
y = iris.target

# tạo scatter plot cho từng cặp feature
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', label=iris.target_names[0])
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', label=iris.target_names[1])
plt.scatter(X[y==2][:, 0], X[y==2][:, 1], color='green', label=iris.target_names[2])

# đặt tên cho trục x và trục y
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# hiển thị legend
plt.legend()

# hiển thị biểu đồ
plt.show()
