import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from K_means_clustering import K_means_clustering
from K_means_clustering import fxdist

# load data
data = loadmat("A2_data.mat")
#print(data.keys())
X_train = data["train_data_01"]# shape 784X12665 i.e 784 features and 12665 samples 
y_train = data["train_labels_01"]
X_test= data["test_data_01"]
y_test=data["test_labels_01"]
#print(y_train.shape)(12665, 2)
# Each image is R^784 ie R^784X1, colon vector, they are stacked after each other. 
# therefore 784X12665 for training 
# Dimension is large since we have 28x28 pixels 

pca = PCA(n_components=2) # We want ti plot it in two dimensions 
Z = pca.fit_transform(X_train.T) # transposed to fit the n_samples and n_features correct 
print(Z.shape) # 12665x2 i.e n_samples*features -> Each row is an imagen in 2 dim
#print(y_train.flatten().shape) (12665,)
Z0 = Z[y_train.flatten() == 0] # y_train gives the right label for the data point, if it is zero we add. 
Z1 = Z[y_train.flatten() == 1]

def plot_PCA():
    plt.figure(figsize=(8,6))
    plt.scatter(Z0[:, 0], Z0[:, 1],marker = 'x', label="Digit 0", alpha=0.6) # Plotting in 2d the two PC values 
    plt.scatter(Z1[:, 0], Z1[:, 1], label="Digit 1", alpha=0.6)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection in 2 dimensions of digits 0,1")
    plt.legend()
    plt.grid(True)
    plt.show()

K = 5
y,c = K_means_clustering(X_train,K)
#print(c.shape) c:(784,K) is the mean of all data points in that cluster 
marker = ['o','x','s','^','v']
plt.figure(figsize=(8,6))

def Kmeans_plot (cluster):
    for k in range(K):
        Zk = Z[y == k] # Making a mask, true for every datapoint where y==k, i.e each datapoint beloing to cluster k. zk only data points for this cluster 

        plt.scatter(Zk[:, 0], Zk[:, 1],marker=marker[k], label=f"Cluster {k+1}", alpha=0.6)

    
    plt.title(f"K-means clusters visualized with PCA. Clusters ={cluster}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

#Kmeans_plot(K)

## Plot for images from centroids ###

def plot_centroids(C, K, title):
    plt.figure(figsize=(10, 3))

    for k in range(K):
        plt.subplot(1, K, k + 1)

        img = C[:, k].reshape(28, 28).T# Since we have done the traspose above 
        plt.imshow(img, cmap='gray')
        plt.title(f"Cluster {k+1}")
        plt.axis('off')

    plt.suptitle(title)
    plt.show()

#plot_centroids(c,K,"Centroid images reconstructed using PCA")

### for exercise 10####

def assign_cluster_labels(y_clusters, true_labels, K):
    cluster_labels = np.zeros(K)

    for k in range(K):
        labels_in_cluster = true_labels[y_clusters == k].flatten()
        cluster_labels[k] = np.bincount(labels_in_cluster).argmax() # fins the largest

    return cluster_labels

def K_means_classifier(x, C, cluster_labels):
    d = fxdist(x, C) # based on the function in K_means _clustering
    k = np.argmin(d)
    return cluster_labels[k]

K = 2
# Train 
y_train_clusters, C = K_means_clustering(X_train, K)
cluster_labels = assign_cluster_labels(y_train_clusters, y_train, K)
y_train_flat = y_train.flatten()
N_train = X_train.shape[1]

# predict train labels using classifier
y_train_pred = np.array([
    K_means_classifier(X_train[:, i], C, cluster_labels)
    for i in range(N_train)
])

print("Training")
for k in range(K):
    mask = (y_train_clusters == k)
    labels_in_cluster = y_train_flat[mask]

    count_0 = np.sum(labels_in_cluster == 0)
    count_1 = np.sum(labels_in_cluster == 1)

    assigned_class = np.bincount(labels_in_cluster).argmax()
    misclassified = np.sum(labels_in_cluster != assigned_class)

    print(f"Cluster {k+1}")
    print(f"# '0': {count_0}")
    print(f"# '1': {count_1}")
    print(f"Assigned class: {assigned_class}")
    print(f"misclassified: {misclassified}")

train_error = np.mean(y_train_pred != y_train_flat)
print("N_train =", N_train)
print("Misclassification rate:", train_error * 100)

# Test 
y_test_flat = y_test.flatten()
N_test = X_test.shape[1]

y_test_clusters = np.array([
    np.argmin(fxdist(X_test[:, i], C))
    for i in range(N_test)
])

y_test_pred = np.array([
    K_means_classifier(X_test[:, i], C, cluster_labels)
    for i in range(N_test)
])

print("Test")
for k in range(K):
    mask = (y_test_clusters == k)
    labels_in_cluster = y_test_flat[mask]

    count_0 = np.sum(labels_in_cluster == 0)
    count_1 = np.sum(labels_in_cluster == 1)

    if len(labels_in_cluster) > 0:
        assigned_class = np.bincount(labels_in_cluster).argmax()
        misclassified = np.sum(labels_in_cluster != assigned_class)
    else:
        assigned_class = None
        misclassified = 0

    print(f"Cluster {k+1}")
    print(f"# '0': {count_0}")
    print(f"# '1': {count_1}")
    print(f"Assigned class: {assigned_class}")
    print(f" misclassified: {misclassified}")

test_error = np.mean(y_test_pred != y_test_flat)
print("\nN_test =", N_test)
print("Misclassification rate:", test_error * 100)