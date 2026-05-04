import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from K_means_clustering import K_means_clustering

# Load data
data = loadmat("A2_data.mat")
print(data.keys())
X_train = data["train_data_01"]# shape 784X12665 i.e 784 features and 12665 samples 
y_train = data["train_labels_01"]
# Each image is R^784 ie R^784X1, colon vector, they are stacked after each other. 
# therefore 784X12665 for training 
# Dimension is large since we have 28x28 pixels 

pca = PCA(n_components=2) # We want ti plot it in two dimensions 
Z = pca.fit_transform(X_train.T) # transposed to fit the n_samples and n_features correct 
print(Z.shape) # 12665x2 i.e n_samples*features -> Each row is an imagen in 2 dim
#print(y_train.flatten().shape) (12665,)
Z0 = Z[y_train.flatten() == 0] # y_train gives the right label for the data point, if it is zero we add. 
Z1 = Z[y_train.flatten() == 1]

plt.figure(figsize=(8,6))
plt.scatter(Z0[:, 0], Z0[:, 1],marker = 'x', label="Digit 0", alpha=0.6) # Plotting in 2d the two PC values 
plt.scatter(Z1[:, 0], Z1[:, 1], label="Digit 1", alpha=0.6)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA projection in 2 dimensions of digits 0,1")
plt.legend()
plt.grid(True)
plt.show()

K = 2
y,c = K_means_clustering(X_train,K)

plt.figure(figsize=(8,6))
marker = ['o','x','s','^','v']
for k in range(K):
    Zk = Z[y == k]
    plt.scatter(Zk[:, 0], Zk[:, 1],marker=marker[k], label=f"Cluster {k+1}", alpha=0.6)

plt.title(f"K-means clusters visualized with PCA. Clusters ={K}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()