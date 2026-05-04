import numpy as np
from sklearn.svm import SVC
from scipy.io import loadmat

data = loadmat("A2_data.mat")
print(data.keys())
X_train = data["train_data_01"]# shape 784X12665 i.e 784 features and 12665 samples 
y_train = data["train_labels_01"] # Shape (12665, 1)
X_test= data["test_data_01"]
y_test=data["test_labels_01"]

D,N_train = X_train.shape
D,N_test= X_test.shape
linearsvm = SVC(kernel='linear')
linearsvm.fit(X_train.T, y_train.flatten())
y_pred_train_linear=linearsvm.predict(X_train.T)
y_pred_test_linear=linearsvm.predict(X_test.T)

## Training data 

def pred_table(y_true, y_pred):
    pred0_true0 = np.sum((y_pred == 0) & (y_true == 0))
    pred0_true1 = np.sum((y_pred == 0) & (y_true == 1))
    pred1_true0 = np.sum((y_pred == 1) & (y_true == 0))
    pred1_true1 = np.sum((y_pred == 1) & (y_true == 1))
    
    return pred0_true0, pred0_true1, pred1_true0, pred1_true1


def print_results(y_train, y_test, y_pred_train, y_pred_test):
    t00, t01, t10, t11 = pred_table(y_train, y_pred_train)
    s00, s01, s10, s11 = pred_table(y_test, y_pred_test)

    N_train = len(y_train)
    N_test = len(y_test)

    train_mis = t01 + t10
    test_mis = s01 + s10

    print("Train")
    print(t00, t01)
    print(t10, t11)
    print("N_train:", N_train)
    print("Misclassified:", train_mis)
    print("Error %:", 100 * train_mis / N_train)

    print("Test")
    print(s00, s01)
    print(s10, s11)
    print("N_test:", N_test)
    print("Misclassified:", test_mis)
    print("Error %:", 100 * test_mis / N_test)
#print_results(y_train.flatten(),y_test.flatten(),y_pred_train_linear, y_pred_test_linear)

## non-linear kernel ####
gamma_values = [ 1e-2,2e-2,3e-2]

for gamma in gamma_values:
    kernelsvm = SVC(kernel='rbf', gamma=gamma) ## change gamma
    kernelsvm.fit(X_train.T, y_train.flatten())

    y_pred_train_kernel=kernelsvm.predict(X_train.T)
    y_pred_test_kernel=kernelsvm.predict(X_test.T)
    print_results(y_train.flatten(),y_test.flatten(),y_pred_train_kernel, y_pred_test_kernel)