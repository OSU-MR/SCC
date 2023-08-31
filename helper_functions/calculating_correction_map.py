
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, eye, vstack, kron, csr_matrix
import numpy as np

def remove_edges(Zi_body_coils,Zi_surface_coils):
    inter_img_body_coils = abs(Zi_body_coils[:,int(Zi_body_coils.shape[1]//4):-int(Zi_body_coils.shape[1]//4)])
    inter_img_body_coils = inter_img_body_coils[:,::-1]

    inter_img_surface_coils = abs(Zi_surface_coils[:,int(Zi_surface_coils.shape[1]//4):-int(Zi_surface_coils.shape[1]//4)])
    inter_img_surface_coils = inter_img_surface_coils[:,::-1]

    return inter_img_body_coils,inter_img_surface_coils

def normalize_images(inter_img_surface_coils,inter_img_body_coils):
    x2d = (inter_img_surface_coils)/np.max((inter_img_surface_coils))
    x3d = (inter_img_body_coils)/np.max((inter_img_body_coils))
    return x2d,x3d


def calculate_correction_map(A,
                             B,
                             lamb=1e-1):
    

    A = csr_matrix(A)
    A = A.toarray().ravel()
    A = diags(A)

    n1 = B.shape[0]  # rows
    n2 = B.shape[1]  # columns

# Operator to take difference across rows
    D1 = diags([np.ones(n2), -np.ones(n2)], [0, 1], shape=(n2, n2))
    D1 = D1.tolil()  # Convert to LIL format for efficient row operations
    # D1[-1, 0] = -1
    D1 = D1[:-1, :]

# Operator to take difference across columns
    D2 = diags([np.ones(n1), -np.ones(n1)], [0, 1], shape=(n1, n1))
    D2 = D2.tolil()  # Convert to LIL format for efficient row operations
    # D2[-1, 0] = -1
    D2 = D2[:-1, :]

# The final operator that computes finite differences
    D = vstack([kron(eye(n1), D1), kron(D2, eye(n2))])

# Reshape the result of the linear system solution
    c = spsolve((A.T @ A) + lamb * (D.T @ D), A.T @ B.flatten()).reshape((n1, n2))
    return A,B,c
