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
        # Step 1: Assign to clusters
        y = []

        # Step 2: Assign new clusters
        C = []

        if fcdist(C, Cold) < conv_tol:
            return y, C

        Cold = C.copy()

    return y, C

def fxdist(x,C):
    # CHANGE
    d = None
    # DO NOT CHANGE
    return d

def fcdist(C1,C2):
    # CHANGE
    d = None
    # DO NOT CHANGE
    return d


def load_data():
    # Replace '/path/to/file/' with the path to your .mat file
    base_path = "/path/to/file/"
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
