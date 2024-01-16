#import pygrappa

import numpy as np
import sigpy as sp
import sigpy.mri as mr
from helper_functions.preprocess import ifftnd, rms_comb
from helper_functions.Interpolation import quaternion_to_directions



# def grappa_reconstruction(ksp,ref_padded):
#     grappa_recon = pygrappa.grappa(ksp,ref_padded,coil_axis=0)
#     grappa_recon = ifftnd(grappa_recon, [1,2])
#     grappa_recon = rms_comb(grappa_recon,0)
#     grappa_img = np.abs(grappa_recon[::-1,grappa_recon.shape[-1]//4:-grappa_recon.shape[-1]//4])
#     grappa_img = grappa_img/np.max(grappa_img)

#     return grappa_img

# def sense_reconstruction(ksp,ref_padded,inversed_correction_map = None,thresh=0.003,crop=0.0):
#     #print the type of ref_padded
#     #print("type(ref_padded)",type(ref_padded))
#     mps = mr.app.EspiritCalib(ref_padded,thresh=thresh,crop=crop).run()#(ref_padded,thresh=0.1,crop=0.50).run() #thresh=0.02,crop=0.05 good
#     #print("mps.shape",mps.shape)
#     if inversed_correction_map is not None:
#         try:
#             mps = np.multiply(mps,inversed_correction_map)
#         except:
#             mps = np.multiply(mps,inversed_correction_map.transpose([1,0]))
#     sense_img = mr.app.SenseRecon(ksp, mps).run()
#     sense_img = complex_image_normalization(sense_img)
#     sense_img = sense_img

#     return sense_img


def sense_reconstruction(ksp,ref_padded,inversed_correction_map = None,thresh=0.003,crop=0, device=0):
    try:
        import cupy
    except:
        print("Cupy is not installed. Please install cupy to use GPU.")
    #check the available GPU
    try:
        with sp.Device(device):
            print("Using GPU")
            # Convert arrays to SigPy arrays, so they are on the proper device
            ksp = sp.to_device(ksp)
            ref_padded = sp.to_device(ref_padded)
    
            mps = mr.app.EspiritCalib(ref_padded, thresh=thresh, crop=crop,device=sp.Device(device)).run()
            
            if inversed_correction_map is not None:
                inversed_correction_map = cupy.asarray(inversed_correction_map)
                mps *= inversed_correction_map

            
            sense_img = mr.app.SenseRecon(ksp, mps,device = sp.Device(device)).run()
            sense_img = complex_image_normalization(sense_img.get())
            sense_img = sense_img
    except Exception as e:
        print("Tried using GPU but encountered this error: ", e)
        print("For using GPU, you need to install specific version of cudatoolkit and cupy.")
        print("please refer to https://github.com/OSU-MR/SCC/blob/main/SCC_env.txt for more information.")
        mps = mr.app.EspiritCalib(ref_padded,thresh=thresh,crop=crop).run()#(ref_padded,thresh=0.1,crop=0.50).run() #thresh=0.02,crop=0.05 good
        if inversed_correction_map is not None:
            mps = np.multiply(mps,inversed_correction_map)
        sense_img = mr.app.SenseRecon(ksp, mps).run()
        sense_img = complex_image_normalization(sense_img)
        sense_img = sense_img

    return sense_img


def complex_image_normalization(img):
    #extract phase and magnitude
    phase = np.angle(img)
    magnitude = np.abs(img)
    #normalize magnitude
    magnitude = magnitude - np.min(magnitude)
    if np.max(magnitude) != 0:
        magnitude = magnitude/np.max(magnitude)
    #reconstruct complex image
    img = magnitude*np.exp(1j*phase)
    return img


def remove_edges(data):
    if data is None:
        return None
    #remove edges [1/4, ... , 1/4]
    data = abs(data[:,int(data.shape[1]//4):-int(data.shape[1]//4)])
    #flip the data to the right direction
    data = data[:,::-1,...]

    return data

import matplotlib.pyplot as plt
def remove_oversampling_phase_direction(data, oversampling_phase_factor = 3):

    if data is None:
        return None
    

    data = data.reshape(oversampling_phase_factor,-1,data.shape[1])


    # # Calculate the sum of the images, ignoring nan values
    summed_image = np.nansum(data, axis=0)

    # Calculate the number of non-nan values for each pixel
    non_nan_count = np.sum(~np.isnan(data), axis=0)

    # Calculate the mean by dividing the summed_image by the number of non-nan values
    data = np.divide(summed_image, non_nan_count, where=non_nan_count!=0)

    return data


def rm_zero_row_col(data_ref,n = None,dim_info_ref = None):
    # Assume that you have a numpy array named `arr`
    if n is None:
        arr = data_ref[:,0,:]
    else:
        arr = data_ref[n,:,0,:]

    mask_nonzero_rows = np.any(arr != 0, axis=1)
    mask_nonzero_cols = np.any(arr != 0, axis=0)

    # Apply the masks to remove zero rows and columns
    for i in range(data_ref.shape[dim_info_ref.index('Cha')]):
        if n is None:
            arr = data_ref[:,i,:]
        else:
            arr = data_ref[n,:,i,:]
        arr_no_zero_rows = arr[mask_nonzero_rows]
        arr_no_zero_rows_or_cols = arr_no_zero_rows[:, mask_nonzero_cols]
        arr_no_zero_rows_or_cols = np.expand_dims(arr_no_zero_rows_or_cols,0)
        if i == 0:
            ref_no_zero = arr_no_zero_rows_or_cols
        
        else:
            ref_no_zero = np.vstack((ref_no_zero,  arr_no_zero_rows_or_cols ))
            #print(ref_no_zero.shape ,arr_no_zero_rows_or_cols.shape)
    return ref_no_zero#.transpose([1,0,2])

def pad_ref(data,data_ref,n,dim_info_ref,dim_info_org):
    
    # print("dim_info_org",dim_info_org)
    # print("data",data.shape)
    # print("dim_info_ref",dim_info_ref)
    # print("data_ref",data_ref.shape)
    if data_ref is None: #no reference data means the data is fully sampled
        ksp = data[n,:,:,:].transpose([1,0,2])
        ref = None
        return ksp,ref

    else:
        try:
            sli_idx = dim_info_org.index('Sli') #see if there is slice dimension
            ksp = data[n,:,:,:].transpose([1,0,2])
            ref = rm_zero_row_col(data_ref,n,dim_info_ref)
            #ref = data_ref[n,0:34,:].transpose([1,0,2])
        except:
            n = None
            ksp = data[:,:,:].transpose([1,0,2])
            ref = rm_zero_row_col(data_ref,n,dim_info_ref)
            #ref = data_ref[0:34,:].transpose([1,0,2])


    # calculate padding
    padding = [(0, 0) if dsize == ref.shape[i] else 
           ((dsize - ref.shape[i]) // 2, 
            (dsize - ref.shape[i]) - (dsize - ref.shape[i]) // 2) 
           for i, dsize in enumerate(ksp.shape)]

    # pad array
    ref_padded = np.pad(ref, padding, mode='constant', constant_values=0)
    return ksp,ref_padded

def rotate_image(xHat, quat):
    """
    rotate_image function rotates the reconstructed image based on a given quaternion.
    
    Parameters:
    xHat (np.ndarray): Reconstructed image
    quat (list): Normalized quaternion (representation of rotation matrix)

    Returns:
    np.ndarray: Rotated reconstructed image
    """
    # Translate quaternion to directions
    ro_dir_vec_tmp, pe_dir_vec_tmp, slc_dir_vec = quaternion_to_directions(quat)

    ro_dir_vec  = -1*ro_dir_vec_tmp
    pe_dir_vec  = -1*pe_dir_vec_tmp

    dir = {'0': 'sag', '1': 'cor', '2': 'tra'}
    dir_in = {'0': 'RL', '1': 'AP', '2': 'FH'}
    I_slc = np.argmax(np.abs(slc_dir_vec))
    slc_dir = dir[str(I_slc)]

    I_ro = np.argmax(np.abs(ro_dir_vec))
    ro_dir = dir_in[str(I_ro)]

    I_pe = np.argmax(np.abs(pe_dir_vec))
    pe_dir = dir_in[str(I_pe)]

    x_dir_vec = ro_dir_vec
    y_dir_vec = pe_dir_vec
    I_x = I_ro
    I_y = I_pe

    if (slc_dir == "cor" or slc_dir == "sag"):
        if (ro_dir != "FH"):
            xHat = np.rot90(xHat)
            x_dir_vec = -1*pe_dir_vec
            y_dir_vec = 1*ro_dir_vec
            I_x = I_pe
            I_y = I_ro

        if x_dir_vec[I_x] > 0:
            xHat = np.flip(xHat, 0)

        if y_dir_vec[I_y] < 0:
            xHat = np.flip(xHat, 1)
            
    elif (slc_dir == "tra"):
        if (ro_dir != "AP"):
            xHat = np.rot90(xHat)
            x_dir_vec = -1*pe_dir_vec
            y_dir_vec = 1*ro_dir_vec
            I_x = I_pe
            I_y = I_ro

        if x_dir_vec[I_x] < 0:
            xHat = np.flip(xHat, 0)

        if y_dir_vec[I_y] < 0:
            xHat = np.flip(xHat, 1)

    
    return xHat
