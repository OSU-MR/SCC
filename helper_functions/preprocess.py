import numpy as np
from numpy import linalg as LA
from numpy.fft import fftshift, ifftshift, ifftn
from numpy.fft import fftshift, ifftshift, fftn
# Chong Chen, Chong.Chen@osumc.edu, 03-29-2023

def adjust_rawdata_dimmension(data, param_org):
    param = param_org[:]
    E0 = param.index('Col')
    E1 = param.index('Lin')
    CHA = param.index('Cha')
    
    dim_add = 0
    num_dim = len(data.shape)
    tmp_dim = param[:]
    
    dims = list(range(num_dim))
    dims.remove(E0)
    dims.remove(E1)
    dims.remove(CHA)
    param.remove('Col')
    param.remove('Lin')
    param.remove('Cha')
    
    if 'Par' not in tmp_dim:
        dim_add += 1
        E2 = num_dim + dim_add - 1
        tmp_dim.append('Par')
    else:
        E2 = tmp_dim.index('Par')
        dims.remove(E2)  
        param.remove('Par')
    
    if 'Sli' not in tmp_dim:
        dim_add += 1
        SLC = num_dim + dim_add - 1
        tmp_dim.append('Sli')
    else:
        SLC = tmp_dim.index('Sli')
        dims.remove(SLC)
        param.remove('Sli')
    
    print(data.shape)
    print(dim_add, len(dims))
    data = np.reshape(data, np.concatenate( (np.array(data.shape), [1]*(dim_add+2-len(dims))), axis = 0) )
    if len(dims) == 0:
        data = np.transpose(data, [E0, E1, E2, CHA, 5,6, SLC])
        tmp_dim[0:7] = ['RO', 'E1', 'E2', 'Cha', 'N', 'S', 'Sli']
    
    if len(dims) == 1:
        N = tmp_dim.index(param[0])
        data = np.transpose(data, [E0, E1, E2, CHA, N, 6, SLC])
        tmp_dim[0:7] = ['RO', 'E1', 'E2', 'Cha', param[0], 'S', 'Sli']
    
    if len(dims) == 2:
        N = tmp_dim.index(param[0])
        S = tmp_dim.index(param[1])
        if data.shape[N] > data.shape[S]:
            data = np.transpose(data, [E0, E1, E2, CHA, N, S, SLC])
            tmp_dim[0:7] = ['RO', 'E1', 'E2', 'Cha', param[0], param[1], 'Sli']
        else:
            print(data.shape)
            print([E0, E1, E2, CHA, N, S, SLC])
            data = np.transpose(data, [E0, E1, E2, CHA, S, N, SLC])
            tmp_dim[0:7] = ['RO', 'E1', 'E2', 'Cha', param[1], param[0], 'Sli']
        
    
    return data, tmp_dim


# remove oversampling along readout
def remove_RO_oversamling(data, axis_RO = 0):
# Remove oversampling RO, axis_RO specify which axis is readout dim (default 0)
    fd = [0.25, 0.25]
    data = np.moveaxis(data, axis_RO, 0)
    N = np.shape(data)
    num_dim = len(N)
    data = np.reshape(data, [N[0], int(np.prod(N)/N[0])])
    # find the acquired k-space lines
    tmp_samp = np.where(np.abs(data[round(N[0]/2), :]) > 0)
    # extract the acquired k-space lines
    I = data[:, tmp_samp]
    # go to image domian
    I = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(
        I, axes=[0]), axes=[0]), axes=[0]) * np.sqrt(N[0])
    # cropping
    I = np.delete(I, range(0, round(N[0]*fd[0]), 1), axis=0)
    I = np.delete(I, np.array(
        range(0, round(N[0]*fd[0]), 1)) + I.shape[0] - round(N[0]*fd[0]), axis=0)
    # go back to k-space domain
    Y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
        I, axes=[0]), axes=[0]), axes=[0]) / np.sqrt(I.shape[0])
    # make sure the unsampled data is zero
    pre_z = np.where(np.abs(data[:,tmp_samp[0][0]]) > 0)[0][0]
    pre_z = int(pre_z*(1-np.sum(fd)))
    Y[:pre_z,:] = 0
    # copy Y to data
    tmp_N = list(N)
    tmp_N[0] = Y.shape[0]
    data_no_os = np.zeros(tmp_N, dtype=np.complex128)
    data_no_os = np.reshape(data_no_os, [tmp_N[0], int(np.prod(tmp_N)/tmp_N[0])])
    data_no_os[:, tmp_samp] = Y
    data_no_os = np.reshape(data_no_os, tmp_N)
    data_no_os = np.moveaxis(data_no_os, 0, axis_RO)
    
    return data_no_os


# noise prewhiting
def comp_noise_prewhitening(noise, axis_ch = 0):
    ## noise; axis_ch specify the cha dim
    noise = np.moveaxis(noise, axis_ch, 0)
    noise = np.reshape(noise, [noise.shape[0], int(np.prod(noise.shape)/noise.shape[0])])
    cov_n = np.matmul(noise, np.matrix.getH(noise))/(noise.shape[1]-1)
    L = np.linalg.cholesky(cov_n)
    nw_mat = np.linalg.inv(L)
    
    return nw_mat

def perf_noise_prewhitening(data, nw_mat, axis_ch = 0):
    data = np.moveaxis(data, axis_ch, 0)
    N_data = data.shape
    data = np.reshape(data, [N_data[0], int(np.prod(N_data)/N_data[0])])
    data = np.matmul(nw_mat, data)
    data = np.reshape(data, N_data)
    data = np.moveaxis(data, 0, axis_ch)
    
    return data


def estimate_noise_from_data(data, axis_spatial = [0,1,2], axis_ch = 3):
    edge_ratio = [8,5,5]
    idx = -1
    for tmp_axis in axis_spatial:
        idx += 1
        data = np.moveaxis(data, tmp_axis, 0)
        N_data = data.shape
        if data.shape[0] > 1:
            data = np.reshape(data, [N_data[0], int(np.prod(N_data)/N_data[0])])
            edge_idx = np.maximum(1,np.minimum(16,round(data.shape[0]/edge_ratio[idx])))
            print(edge_idx)
            tmp1 = data[0:edge_idx,:]
            tmp2 = data[-1*edge_idx:,:]
            data = np.concatenate((tmp1, tmp2) , axis=0)
            data = np.reshape(data, [data.shape[0]] + list(N_data[1:]))
        data = np.moveaxis(data, 0, tmp_axis)
    data = np.moveaxis(data,axis_ch,0)
    data = np.reshape(data, [data.shape[0], int(np.prod(data.shape)/data.shape[0])])

    CH = data.shape[0]
    n_std = np.zeros([CH])
    for ch_idx in range(0,CH,1):
        data_ch = data[ch_idx,:]
        n_std[ch_idx] =  np.std(data_ch[np.abs(data_ch) != 0], axis=0)
        
    return n_std



# coil compression using PCA
#from numpy import linalg as LA
def compress_data_with_pca(data, channel_keep, axis_CH = 0, extra_data = None, extra_axis_CH = 0, extra_data2 = None, extra_axis_CH2 = 0):
    # input: data; channel_keep (number of virtual channels kept); axis_CH specify the coil dim
    # PCA coil compression
    if channel_keep is None: 
        #print with yellow color
        print('\033[93m'+'Skipping coil compression since the channel_keep is not specified'+'\033[0m')
        return data, extra_data, extra_data2, None
    elif data.shape[axis_CH] <= channel_keep:
        print('\033[91m'+'Warning, the number of channels is equal or less than the number of channels kept.\n Check your "channel_keep" parameters is you want to do coil compression'+'\033[0m')
        return data, extra_data, extra_data2, None
    else:
        print('compress ',data.shape[axis_CH],' to ',channel_keep,' channels')
    data = np.moveaxis(data, axis_CH, 0)
    N_data = data.shape
    data = np.reshape(data, [N_data[0], int(np.prod(N_data)/N_data[0])])
    data = np.transpose(data, [1, 0])
    cov_matrix = np.matmul(np.matrix.getH(data),data)
    u,s,vh = LA.svd(cov_matrix)
    data_compressed = preform_compressing(data, channel_keep, axis_CH, N_data, u)
    if extra_data is None:    
        return data_compressed, u
    elif extra_data2 is None:
        extra_data_compressed = preform_compressing(extra_data, channel_keep, extra_axis_CH, N_data, u, reshape_flag = True)
        return data_compressed, extra_data_compressed, u
    else:
        extra_data_compressed  = preform_compressing(  extra_data, channel_keep,  extra_axis_CH, N_data, u, reshape_flag = True)
        extra_data2_compressed = preform_compressing( extra_data2, channel_keep, extra_axis_CH2, N_data, u, reshape_flag = True)
        return data_compressed, extra_data_compressed, extra_data2_compressed, u

def preform_compressing(data, channel_keep, axis_CH, N_data, u, reshape_flag = False):
    if reshape_flag:
        print('compress ',data.shape[axis_CH],' to ',channel_keep,' channels')
        data = np.moveaxis(data, axis_CH, 0)
        N_data = data.shape
        data = np.reshape(data, [N_data[0], int(np.prod(N_data)/N_data[0])])
        data = np.transpose(data, [1, 0])

    #print("data.shape:",data.shape)
    #print("u: ",u.shape)
    data = np.matmul(data, u)
    data_compressed = data[:,0:channel_keep]
    data_compressed = np.transpose(data_compressed, [1,0])
    data_compressed = np.reshape(data_compressed, [channel_keep, int(np.prod(N_data)/N_data[0])])
    data_compressed = np.reshape(data_compressed, list([channel_keep]) + list(N_data[1:]))
    data_compressed = np.moveaxis(data_compressed, 0, axis_CH)
    return data_compressed




# average data
def average_data(data, axis_avg):
    # average data along axis_avg
    samp = np.abs(data) > 0
    samp = samp.astype(int)
    data_avg = np.sum(data, axis = axis_avg) / (np.sum(samp, axis = axis_avg) + np.finfo(float).eps)
    
    return data_avg


# not used anymore
def remove_RO_oversamling_not_used(data):
    # data: first dimension should be Readout dimension
    
    fd = [0.25, 0.25]
    N = np.shape(data)
    I = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(data, axes=[0]), axes=[0]), axes=[0]) * np.sqrt(N[0])
    I = np.delete(I, range(0,round(N[0]*fd[0]),1),axis = 0)
    I = np.delete(I, np.array(range(0,round(N[0]*fd[0]),1)) + I.shape[0] - round(N[0]*fd[0]), axis = 0)
    Y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(I, axes=[0]), axes=[0]), axes=[0]) / np.sqrt(I.shape[0]) 
    RO = np.shape(data)[0]
    samp = np.sum(np.abs(np.reshape(data, [RO, np.prod(N[1:])])), axis=1) != 0
    # parameter related to assymetry echo
    pre_z = np.where(samp == True)[0][0]
    pre_z = int(pre_z*(1-np.sum(fd)))
    # make sure the unsampled data is zero
    Y = np.delete(Y, range(0,pre_z,1),axis = 0)
    num_dim = len(Y.shape)
    padsize = [(0, 0)]*num_dim
    padsize[0] =  (pre_z, 0)
    Y = np.pad(Y, padsize)
    print(Y.shape)
    print(data.shape)
    
    return Y


def adjust_rawdata_dimmension_not_used(data, param_org):
    param = param_org[:]
    E0 = param.index('Col')
    E1 = param.index('Lin')
    CHA = param.index('Cha')
    
    dim_add = 0
    num_dim = len(data.shape)
    tmp_dim = param[:]
    
    dims = list(range(num_dim))
    dims.remove(E0)
    dims.remove(E1)
    dims.remove(CHA)
    param.remove('Col')
    param.remove('Lin')
    param.remove('Cha')
    
    if 'Par' not in tmp_dim:
        dim_add += 1
        E2 = num_dim + dim_add - 1
        tmp_dim.append('Par')
    else:
        E2 = tmp_dim.index('Par')
        dims.remove(E2)  
        param.remove('Par')
    
    if 'Phs' not in tmp_dim:
        dim_add += 1
        PHS = num_dim + dim_add - 1
        tmp_dim.append('Phs')
    else:
        PHS = tmp_dim.index('Phs')
        dims.remove(PHS)
        param.remove('Phs')
    
    if 'Set' not in tmp_dim:
        dim_add += 1
        SET = num_dim + dim_add - 1
        tmp_dim.append('Set')
    else:
        SET = tmp_dim.index('Set')
        dims.remove(SET)
        param.remove('Set')
    
    if 'Sli' not in tmp_dim:
        dim_add += 1
        SLC = num_dim + dim_add - 1
        tmp_dim.append('Sli')
    else:
        SLC = tmp_dim.index('Sli')
        dims.remove(SLC)
        param.remove('Sli')
        
    data = np.reshape(data, np.concatenate( (np.array(data.shape), [1]*dim_add), axis = 0) )
    data = np.transpose(data, [E0, E1, E2, CHA, PHS, SET, SLC] + dims)

    tmp_dim[-len(dims):] = param
    tmp_dim[0:7] = ['RO', 'E1', 'E2', 'Cha', 'Phs', 'Set', 'Sli']
    return data, tmp_dim


# some useful functiionos
def ifftnd(kspace, axes=[-1]):
#    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def fftnd(img, axes=[-1]):
#    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(abs(sig)**2, axis))
