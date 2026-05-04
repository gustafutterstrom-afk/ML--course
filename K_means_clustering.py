import numpy as np
import scipy

def K_means_clustering(X, K):
    """
    Perform K-means clustering on input data.

    Parameters:
    - X: numpy.ndarray
        DxN matrix of input data.
    - K: int
        Number of clusters.

    Returns:
    - y: numpy.ndarray
        Nx1 vector of cluster assignments.
    - C: numpy.ndarray
        DxK matrix of cluster centroids.
    """

    D, N = X.shape

    intermax = 50
    conv_tol = 1e-6

    # Initialize
    C = np.mean(X, axis=1).reshape(D, 1) + np.std(X, axis=1).reshape(D, 1) * np.random.randn(D, K)
    y = np.zeros(N)
    Cold = C.copy()

    for i in range(intermax):
    # Step 1
        y = step_assign_cluster(X, C)
    # Step 2
        C_new, movement = step_compute_mean(X, y, C)
        C = C_new

        if movement < conv_tol:
            break
    return y, C

def step_assign_cluster(X, C):

    N = X.shape[1]
    y = np.zeros(N, dtype=int) 

    for n in range(N):
        d = fxdist(X[:, n], C)
        y[n] = np.argmin(d)# gives the index where the samllest index is. d is (K,) i.e index of cluster

    return y

def step_compute_mean(X, y, C):
    D, K = C.shape
    C_new = np.zeros((D, K))

    for k in range(K):
        cluster_points = X[:, y == k]

        if cluster_points.shape[1] > 0:
            C_new[:, k] = np.mean(cluster_points, axis=1)
        else:
            C_new[:, k] = C[:, k]  # keep old centroid

    movement = np.max(fcdist(C_new, C))

    return C_new, movement

def fxdist(x,C):
    # CHANGE
    d = np.linalg.norm(C-x[:,None],axis=0) # since x:(D,) and C:(D,K) 
    # DO NOT CHANGE
    return d

def fcdist(C1,C2):
    # CHANGE
    d = np.linalg.norm(C1-C2,axis=0)#since C:(D,K) and C:(D,K) d:(K,)
    # DO NOT CHANGE
    return d


def load_data():
    # Replace '/path/to/file/' with the path to your .mat file
    base_path = "/Users/gustafutterstrom/Desktop/Utbildning/Progg/ClonedRepos/ML--course/"
    mat_file_path = base_path + "A2_data.mat"
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: File '{mat_file_path}' not found.")
        mat_data = None

    if mat_data is not None:
        # Access variables from the .mat file
        test_data = mat_data['test_data_01']
        test_labels = mat_data['test_labels_01']
        train_data = mat_data['train_data_01']
        train_labels = mat_data['train_labels_01']
        return [test_data, test_labels, train_data, train_labels]

if __name__ == "__main__":
    data = load_data()
    nbr_clusters = 4 # Replace with you chosen int
    y, C = K_means_clustering(data[2], nbr_clusters)
