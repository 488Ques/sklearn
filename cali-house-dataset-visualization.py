import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the dataset
data = fetch_california_housing()

# Create a scatter plot of latitude vs. longitude
plt.scatter(data.data[:, 0], data.data[:, 1], c=data.target)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Californian Housing Prices')
plt.colorbar()
plt.show()