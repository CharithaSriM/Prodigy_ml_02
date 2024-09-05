import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Sample dataset
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8,9,10],
    'Annual_Spend': [5000, 7000, 8000, 4000, 6500, 3000, 7200, 5800,8200,6100],
    'Frequency_of_Purchases': [12, 20, 15, 10, 14, 8, 22, 16, 18, 12],
    'Average_Purchase_Value': [400, 350, 533, 400, 464, 375, 327, 362, 456, 508]
}
df = pd.DataFrame(data)
#preparing the data
X = df.drop('CustomerID', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#determining the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)
''