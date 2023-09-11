# %%
import numpy as np
# Q1 Part 2


# This function takes Matrix a and a number k and conduct the algorithm for k times.
def triangulation(a, k):
    b = a
    for i in range(k):
        q, r = np.linalg.qr(b)
        b = r @ q
    return b


# This function take a triangular matrix 'a' and return its sorted eigenvalues. (diag of matrix)
def eigenvalues(a):
    return -1*np.sort(-1*(np.diag(a)))


# This function take a matrix 'a' and one of its eigenvalues and return its eigenvector. (last row of vh)
def eigenvector(a, lambda_i):
    u, s, vh = np.linalg.svd(a-(lambda_i*np.identity(a.shape[0])))
    return vh[a.shape[1]-1, :]


# This function create the matrix of eigenvectors of matrix 'a'.
def eigenvector_matrix(a, k):
    v = np.zeros((a.shape[0], a.shape[0]))
    for i in range(a.shape[0]):
        v[:, i] = eigenvector(a, eigenvalues(triangulation(a, k))[i]).T
    return v


# This function take the matrix 'a' and print its eigenvalues and eigenvectors.
def eigen_pairs(a, k):
    print(eigenvalues(triangulation(a, k)))
    print(eigenvector_matrix(a, k))
