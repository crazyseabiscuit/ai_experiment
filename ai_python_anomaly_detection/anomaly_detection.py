
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('NAB_dataset.csv', parse_dates=['timestamp'], index_col='timestamp')

# Reshape data for K-Means
data = df[['value']]

# Initialize and fit the K-Means model
# We'll use a small number of clusters to group the normal data.
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(data)

# Calculate the distance of each point to its assigned cluster center
distances = []
for i, c in enumerate(kmeans.cluster_centers_):
    # Get all points belonging to cluster i
    cluster_points = data[df['cluster'] == i]
    # Calculate distance from each point to the cluster center
    distance = np.abs(cluster_points - c)
    distances.append(distance)

# Combine the distances back into a single series, aligned with the original dataframe
df['distance_to_center'] = pd.concat(distances).sort_index()

# Define a threshold for anomalies. For example, any point where the distance
# is greater than the 99th percentile of all distances can be considered an anomaly.
anomaly_threshold = df['distance_to_center'].quantile(0.99)
df['anomaly'] = df['distance_to_center'] > anomaly_threshold

anomalies = df[df['anomaly'] == True]

# Visualize the results
plt.figure(figsize=(15, 7))
# Plot normal data points, colored by their cluster
plt.scatter(df[df['anomaly'] == False].index, df[df['anomaly'] == False]['value'], c=df[df['anomaly'] == False]['cluster'], cmap='viridis', label='Normal Data by Cluster', s=10)
# Plot anomalies
plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomaly', s=50, marker='x')
# Plot cluster centers
plt.scatter(df.index, kmeans.cluster_centers_[df['cluster']], color='black', s=100, alpha=0.2, marker='o', label='Cluster Centers (projected)')


plt.title('Anomaly Detection using K-Means Clustering')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('anomaly_detection_plot_kmeans.png')
plt.show()

print("Anomaly detection complete. Plot saved to anomaly_detection_plot_kmeans.png")
