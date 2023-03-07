import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('clustering/customer_behavior.csv')

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['number_of_orders', 'payment_amount']])

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)

# Add cluster information to original DataFrame
df['cluster'] = kmeans.labels_

# Get cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Visualize clusters and cluster centers

plt.hist(df['payment_amount'], bins=20)

# Đặt tên cho trục x
plt.xlabel('Payment Amount')

# Đặt tên cho trục y
plt.ylabel('Frequency')

# Hiển thị biểu đồ
plt.show()

plt.scatter(df['number_of_orders'], df['payment_amount'], c=df['cluster'], alpha=0.5)
plt.scatter(centers[:,0], centers[:,1], marker='o', s=200, c='red')

plt.xlabel('Number of Orders')
plt.ylabel('Payment Amount')
plt.title('K-means Clustering with Cluster Centers')
plt.show()