
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import diags, eye, vstack, kron, csr_matrix, spdiags
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal

def remove_edges(Zi_body_coils,Zi_surface_coils):
    inter_img_body_coils = abs(Zi_body_coils[:,int(Zi_body_coils.shape[1]//4):-int(Zi_body_coils.shape[1]//4)])
    inter_img_body_coils = inter_img_body_coils[:,::-1]

    inter_img_surface_coils = abs(Zi_surface_coils[:,int(Zi_surface_coils.shape[1]//4):-int(Zi_surface_coils.shape[1]//4)])
    inter_img_surface_coils = inter_img_surface_coils[:,::-1]

    return inter_img_body_coils,inter_img_surface_coils

def normalize_image(A):
    return A/np.max(A)


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
    D1 = D1[:-1, :]

# Operator to take difference across columns
    D2 = diags([np.ones(n1), -np.ones(n1)], [0, 1], shape=(n1, n1))
    D2 = D2.tolil()  # Convert to LIL format for efficient row operations
    D2 = D2[:-1, :]

# The final operator that computes finite differences
    D = vstack([kron(eye(n1), D1), kron(D2, eye(n2))])

# Reshape the result of the linear system solution
    c = spsolve((A.T @ A) + lamb * (D.T @ D), A.T @ B.flatten()).reshape((n1, n2))
    return A,B,c


def window_creator(target_len = (128,128,128), window_len = (128,32,32), alpha = 0.5):

    #create alphas
    min_window_len = np.min(window_len)
    alphas = (min_window_len/window_len)*alpha

    #set the alpha to 0 if the window length is equal to the target length
    shared_indices = [i for i, (t, w) in enumerate(zip(target_len, window_len)) if t == w]
    alphas[shared_indices] = 0

    # Create a 1D Tukey window for x, y, and z
    tukey_1D_x = signal.windows.tukey(window_len[0], alpha=alphas[0])
    tukey_1D_y = signal.windows.tukey(window_len[1], alpha=alphas[1])
    tukey_1D_z = signal.windows.tukey(window_len[2], alpha=alphas[2])

    # Create a 3D Tukey window by taking the outer product of the 1D window
    tukey_3D = tukey_1D_x[:, np.newaxis, np.newaxis] * tukey_1D_y[np.newaxis, :, np.newaxis] * tukey_1D_z[np.newaxis, np.newaxis, :]
    

    #extending the window to 128x128x128
    pad_len_x = (target_len[0]-window_len[0])//2
    pad_len_y = (target_len[1]-window_len[1])//2
    pad_len_z = (target_len[2]-window_len[2])//2
    tukey_3D = np.pad(tukey_3D, ((pad_len_x, pad_len_x), (pad_len_y, pad_len_y), (pad_len_z, pad_len_z)), 'constant', constant_values=(0, 0))

    return tukey_3D



def extendTo128x64x64(A, target_shape = (128, 64, 64), final_target_shape = (128,64,64) , alpha = 0.5):
    '''
    We need to extend the matrix to 128x64x64 to keep the voxel size cubic , 
    then apply the window, 
    then extend to 128x64x64
    '''
    #print(A.shape) #(128, 32, 32)
    x, y, z = A.shape

    if x == target_shape[0] and y == target_shape[1] and z == target_shape[2]:
        print("The matrix is already ", target_shape)
        return A
    else:

        # do fft to the input A
        A = np.fft.fftn(A, axes=(0,1,2))
        A = np.fft.fftshift(A)

        # Calculate the padding sizes for each dimension
        pad_sizes = [(int(np.ceil((target - original) / 2)), int(np.floor((target - original) / 2))) for target, original in zip(target_shape, A.shape)]

        # Apply padding
        A_padded = np.pad(A, pad_sizes, mode='constant', constant_values=0)

        #apply window
        tukey_3D = window_creator(target_shape, A.shape, alpha = alpha)
        A_padded = A_padded*tukey_3D

        # do ifft to the padded A
        A_padded = np.fft.ifftshift(A_padded)
        A_padded = np.fft.ifftn(A_padded, axes=(0,1,2))

        #zero padding to final target shape
        pad_sizes = [(int(np.ceil((target - original) / 2)), int(np.floor((target - original) / 2))) for target, original in zip(final_target_shape, A_padded.shape)]
        A_padded = np.pad(A_padded, pad_sizes, mode='constant', constant_values=0)

        # Check the shape of the padded array
        print("The 3D reference matrix has been extended to", A_padded.shape)
        return A_padded
    
def extendTo128x128x128(A, target_shape = (128, 64, 64), final_target_shape = (128,128,128) , alpha = 0.5):
    '''
    We need to extend the matrix to 128x64x64 to keep the voxel size cubic , 
    then apply the window, 
    then extend to 128x64x64
    '''
    #print(A.shape) #(128, 32, 32)
    x, y, z = A.shape

    if x == target_shape[0] and y == target_shape[1] and z == target_shape[2]:
        print("The matrix is already ", target_shape)
        return A
    else:

        # do fft to the input A
        A = np.fft.fftn(A, axes=(0,1,2))
        A = np.fft.fftshift(A)

        # Calculate the padding sizes for each dimension
        pad_sizes = [(int(np.ceil((target - original) / 2)), int(np.floor((target - original) / 2))) for target, original in zip(target_shape, A.shape)]

        # Apply padding
        A_padded = np.pad(A, pad_sizes, mode='constant', constant_values=0)

        #apply window
        tukey_3D = window_creator(target_shape, A.shape, alpha = alpha)
        A_padded = A_padded*tukey_3D

        # do ifft to the padded A
        A_padded = np.fft.ifftshift(A_padded)
        A_padded = np.fft.ifftn(A_padded, axes=(0,1,2))

        #zero padding to final target shape
        pad_sizes = [(int(np.ceil((target - original) / 2)), int(np.floor((target - original) / 2))) for target, original in zip(final_target_shape, A_padded.shape)]
        A_padded = np.pad(A_padded, pad_sizes, mode='constant', constant_values=0)

        # Check the shape of the padded array
        print("The 3D reference matrix has been extended to", A_padded.shape)
        return A_padded

def reshapeTo64x64x64(A, target_shape = (64, 64, 64), alpha = 0.5):
    #print(A.shape) #(128, 32, 32)
    x, y, z = A.shape

    if x == target_shape[0] and y == target_shape[1] and z == target_shape[2]:
        print("The matrix is already ", target_shape)
        return A
    else:
        #cut the x dimension to 64
        len2cut = (x - target_shape[0])//2
        A = A[len2cut:-len2cut,:,:]

        # do fft to the input A
        A = np.fft.fftn(A, axes=(0,1,2))
        A = np.fft.fftshift(A)

        # Calculate the padding sizes for each dimension
        pad_sizes = [(int(np.ceil((target - original) / 2)), int(np.floor((target - original) / 2))) for target, original in zip(target_shape, A.shape)]

        # Apply padding
        A_padded = np.pad(A, pad_sizes, mode='constant', constant_values=0)

        #apply window
        tukey_3D = window_creator(target_shape, A.shape, alpha = alpha)
        A_padded = A_padded*tukey_3D

        # do ifft to the padded A
        A_padded = np.fft.ifftshift(A_padded)
        A_padded = np.fft.ifftn(A_padded, axes=(0,1,2))

        # Check the shape of the padded array
        print("The 3D reference matrix has been reshaped to", A_padded.shape)
        return A_padded

def fft_cropping_from128(A, target_shape = (128, 32, 32)):
    if A.shape[1] == 128 and A.shape[2] == 128:
        A = A[:,32:-32,32:-32]
    A = np.fft.fftn(A, axes=(0,1,2))
    A = np.fft.fftshift(A)
    #calculate the cropping size
    x, y, z = A.shape
    x2cut = (x - target_shape[0])//2
    y2cut = (y - target_shape[1])//2
    z2cut = (z - target_shape[2])//2
    #crop the matrix
    if x2cut > 0:
        A = A[x2cut:-x2cut,:,:]
    elif x2cut < 0:
        #we need to pad the matrix
        pad_len = -x2cut
        A = np.pad(A, ((pad_len, pad_len), (0, 0), (0, 0)), 'constant', constant_values=(0, 0)) 
    else:
        pass

    if y2cut > 0:
        A = A[:,y2cut:-y2cut,:]
    elif y2cut < 0:
        #we need to pad the matrix
        pad_len = -y2cut
        A = np.pad(A, ((0, 0), (pad_len, pad_len), (0, 0)), 'constant', constant_values=(0, 0))
    else:
        pass

    if z2cut > 0:
        A = A[:,:,z2cut:-z2cut]
    elif z2cut < 0:
        #we need to pad the matrix
        pad_len = -z2cut
        A = np.pad(A, ((0, 0), (0, 0), (pad_len, pad_len)), 'constant', constant_values=(0, 0))
    else:
        pass

    A = np.fft.ifftshift(A)
    A = np.fft.ifftn(A, axes=(0,1,2))
    print("The 3D reference matrix has been cropped to", A.shape)
    return A


def fft_cropping_from64(A, target_shape = (128, 32, 32)):

    #calculate the cropping size
    x, y, z = A.shape
    x2cut = (x - target_shape[0])//2
    y2cut = (y - target_shape[1])//2
    z2cut = (z - target_shape[2])//2
    #
    #print(x2cut, y2cut, z2cut)
    #crop the matrix in k-space
    A = np.fft.fftn(A, axes=(0,1,2))
    A = np.fft.fftshift(A)
    if x2cut > 0:
        A = A[x2cut:-x2cut,:,:]
    if y2cut > 0:
        A = A[:,y2cut:-y2cut,:]
    if z2cut > 0:
        A = A[:,:,z2cut:-z2cut]
    A = np.fft.ifftshift(A)
    A = np.fft.ifftn(A, axes=(0,1,2))

    #pad the matrix in image space
    if x2cut < 0:
        #we need to pad the matrix
        pad_len = -x2cut
        A = np.pad(A, ((pad_len, pad_len), (0, 0), (0, 0)), 'constant', constant_values=(0, 0)) 
    if y2cut < 0:
        #we need to pad the matrix
        pad_len = -y2cut
        A = np.pad(A, ((0, 0), (pad_len, pad_len), (0, 0)), 'constant', constant_values=(0, 0))
    if z2cut < 0:
        #we need to pad the matrix
        pad_len = -z2cut
        A = np.pad(A, ((0, 0), (0, 0), (pad_len, pad_len)), 'constant', constant_values=(0, 0))

    print("The 3D reference matrix has been cropped to", A.shape)
    return A


# def fft_cropping(A, target_shape = (128, 32, 32)):
#     A = np.fft.fftn(A, axes=(0,1,2))
#     A = np.fft.fftshift(A)
#     #calculate the cropping size
#     x, y, z = A.shape
#     x2cut = (x - target_shape[0])//2
#     y2cut = (y - target_shape[1])//2
#     z2cut = (z - target_shape[2])//2
#     #crop the matrix
#     if x2cut > 0:
#         A = A[x2cut:-x2cut,:,:]
#     elif x2cut < 0:
#         #we need to pad the matrix
#         pad_len = -x2cut
#         A = np.pad(A, ((pad_len, pad_len), (0, 0), (0, 0)), 'constant', constant_values=(0, 0)) 
#     else:
#         pass

#     if y2cut > 0:
#         A = A[:,y2cut:-y2cut,:]
#     elif y2cut < 0:
#         #we need to pad the matrix
#         pad_len = -y2cut
#         A = np.pad(A, ((0, 0), (pad_len, pad_len), (0, 0)), 'constant', constant_values=(0, 0))
#     else:
#         pass

#     if z2cut > 0:
#         A = A[:,:,z2cut:-z2cut]
#     elif z2cut < 0:
#         #we need to pad the matrix
#         pad_len = -z2cut
#         A = np.pad(A, ((0, 0), (0, 0), (pad_len, pad_len)), 'constant', constant_values=(0, 0))
#     else:
#         pass

#     A = np.fft.ifftshift(A)
#     A = np.fft.ifftn(A, axes=(0,1,2))
#     print("The 3D reference matrix has been cropped to", A.shape)
#     return A


def calculate_correction_map_3D(x3_s_in, x3_b_in, lamb = 1e-3, tol=1e-4, maxiter=500, sensitivity_correction_maps=False, debug = False):
    #save x3_s_in and x3_b_in in npy files
    #np.save('x3_s_in.npy', x3_s_in)
    #np.save('x3_b_in.npy', x3_b_in)

    x3_s_in = np.abs(extendTo128x64x64(x3_s_in))
    x3_b_in = np.abs(extendTo128x64x64(x3_b_in))
    #x3_s_in = extendTo128x128x128(x3_s_in)
    #x3_b_in = extendTo128x128x128(x3_b_in)

    #x3_s_in = reshapeTo64x64x64(x3_s_in)
    #x3_b_in = reshapeTo64x64x64(x3_b_in)

    #x3_s = x3_s_in/np.max(x3_b_in)
    #x3_b = x3_b_in/np.max(x3_b_in)    

    if sensitivity_correction_maps:
        x3_s = x3_s_in/np.max(x3_b_in)
        x3_b = x3_b_in/np.max(x3_b_in)
        #exchange x3_s and x3_b for finding the sensitivity correction map
        temp = x3_s
        x3_s = x3_b
        x3_b = temp
    else:
        x3_s = x3_s_in/np.max(x3_s_in)
        x3_b = x3_b_in/np.max(x3_s_in)
        
    # plt.figure(figsize=(5, 10))
    # plt.subplot(2, 2, 1)
    # plt.imshow(np.abs(np.squeeze(x3_s[:, 32, :]))**1, cmap='gray')
    # plt.axis('off')
    # plt.colorbar()
    # plt.title('surface coil')

    # plt.subplot(2, 2, 2)
    # plt.imshow(np.abs(np.squeeze(x3_b[:, 32, :]))**1, cmap='gray')
    # plt.axis('off')
    # plt.colorbar()
    # plt.title('body coil')

    

    if debug:
        print("*****************start********************")
        print(np.max(x3_s_in), np.max(x3_b_in))
        print(np.max(x3_s), np.max(x3_b))
        print("******************end*******************")
        print("Shape of x3_s:", x3_s.shape)
        print("Shape of x3_b:", x3_b.shape)

    assert x3_s.shape == x3_b.shape, "x3_s and x3_b must have the same shape"

    # Parameters
    n = x3_s.shape#[64, 32, 32]
    nx, ny, nz = n

    # Start timing
    start_time = time.time()

    # Convert x3_s into a sparse operator
    A = spdiags(x3_s.flatten(order='F'), 0, np.prod(n), np.prod(n))  #A = diags(x3_s.flatten())

    # Operator to take difference across x-axis (rows)
    D1 = diags([np.ones(nx), -np.ones(nx)], [0, 1], shape=(nx, nx))
    # Operator to take difference across y-axis (columns)
    D2 = diags([np.ones(ny), -np.ones(ny)], [0, 1], shape=(ny, ny))
    # Operator to take difference across z-axis (depth)
    D3 = diags([np.ones(nz), -np.ones(nz)], [0, 1], shape=(nz, nz))

    D1 = D1.tolil()  # Convert to LIL format for efficient row operations
    D2 = D2.tolil()
    D3 = D3.tolil()

    # D1 = D1[:-1, :]  # Remove the last row
    # D2 = D2[:-1, :]
    # D3 = D3[:-1, :]
    D1[-1,0] = -1  
    D2[-1,0] = -1
    D3[-1,0] = -1

    D1 = D1.tocsr()  # Convert back to CSR format
    D2 = D2.tocsr()
    D3 = D3.tocsr()



    # The final operator that computes finite differences
    D = vstack([
        kron(kron(eye(nz), eye(ny)), D1),
        kron(kron(eye(nz), D2), eye(nx)),
        kron(kron(D3, eye(ny)), eye(nx))
    ])

    # Regularized least-squares solution
    B = (A.T @ A + lamb * D.T @ D)
    #c, _ = cg(B, A.T @ x3_b.flatten(order='F'), tol=tol, maxiter=maxiter)
    c = myCGfun(B, A.T @ x3_b.flatten(order='F'), tol=tol, maxiter=maxiter)
    c = c.reshape((nx, ny, nz),order='F')

    x3_sc = c * x3_s

    print(f"Time to find 3D correction map is: {time.time() - start_time:.3f} s")
    
    if debug:
        # Print time
        
        # Plotting
        plt.figure(figsize=(5, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(np.squeeze(x3_s[:, ny//2, :]), cmap='gray')
        plt.axis('off')
        plt.title('surface coil')

        plt.subplot(2, 2, 2)
        plt.imshow(np.squeeze(x3_b[:, ny//2, :]), cmap='gray')
        plt.axis('off')
        plt.title('body coil')

        plt.subplot(2, 2, 3)
        plt.imshow(np.squeeze(c[:, ny//2, :]), cmap='jet')
        plt.axis('off')
        plt.title('correction map')
        plt.colorbar(shrink=0.7)

        plt.subplot(2, 2, 4)
        plt.imshow(np.squeeze(x3_sc[:, ny//2, :]), cmap='gray')
        plt.axis('off')
        plt.title('corrected')

        plt.show()

    #return fft_cropping_from64(c)
    return fft_cropping_from64(c)


# def myCGfun(B, y, tol, maxiter):
#     try:
#         m = len(y)
#     except:
#         m = y.shape[0]
#     x = np.zeros(m)
#     r = y - csr_matrix(B @ x).T
#     x = csr_matrix(np.zeros(m)).T
#     p = r
#     r2old = r.T @ r
    
#     for iter in range(maxiter):
#         Bp = B @ p
#         alpha = complex(r2old / (p.T @ Bp))
#         x = x + alpha * p
#         r = r - alpha * Bp
#         r2new = r.T @ r
        
#         if np.sqrt(r2new) >= tol:
#             ...
#         else:
#             print(f'CG converged after {iter+1} iterations.')
#             return x
        
#         p = r + complex(r2new / r2old) * p
#         r2old = r2new
    
#     print('CG reached max iterations without converging.')
#     return x


# Custom CG function
def myCGfun(B, y, tol, maxiter):
    m = len(y)
    x = np.zeros(m)
    r = y - B @ x
    p = r
    r2old = r.T @ r
    
    for iter in range(maxiter):
        Bp = B @ p
        alpha = r2old / (p.T @ Bp)
        x = x + alpha * p
        r = r - alpha * Bp
        r2new = r.T @ r
        
        if np.sqrt(r2new) < tol:
            print(f'CG converged after {iter+1} iterations.')
            return x
        
        p = r + (r2new / r2old) * p
        r2old = r2new
    
    print('CG reached max iterations without converging.')
    return x