import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering

n = 2000
r1 = 5 + np.random.randn(n)
r2 = 10 + np.random.randn(n)
alpha1 = np.random.rand(n) * np.pi * 2
alpha2 = np.random.rand(n) * np.pi * 2
plt.plot(r1*np.sin(alpha1), r1*np.cos(alpha1), 'ob')
plt.plot(r2*np.sin(alpha2), r2*np.cos(alpha2), '+r')
plt.show()




# Step 1: Generate synthetic data with clusters of different densities
np.random.seed(42)

# Dense cluster
X1 = np.random.normal(loc=[0, 0], scale=0.1, size=(200, 2))
# Sparse cluster
X2 = np.random.normal(loc=[3, 3], scale=1.0, size=(200, 2))
# Combine clusters
X = np.vstack((X1, X2))

# Step 2: Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Step 3: Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral.fit_predict(X)

# Step 4: Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data
axes[0].scatter(X[:, 0], X[:, 1], c='gray', s=30)
axes[0].set_title('Original Data')

# K-Means result
axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=30)
axes[1].set_title('K-Means Clustering')

# Spectral Clustering result
axes[2].scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', s=30)
axes[2].set_title('Spectral Clustering')

plt.tight_layout()
plt.show()
