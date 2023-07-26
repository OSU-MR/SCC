import pygrappa
import numpy as np
import sigpy.mri as mr
from brightness_correction.preprocess import ifftnd, rms_comb
from brightness_correction.Interpolation import quaternion_to_directions



def grappa_reconstruction(ksp,ref_padded):
    grappa_recon = pygrappa.grappa(ksp,ref_padded,coil_axis=0)
    grappa_recon = ifftnd(grappa_recon, [1,2])
    grappa_recon = rms_comb(grappa_recon,0)
    grappa_img = np.abs(grappa_recon[::-1,grappa_recon.shape[-1]//4:-grappa_recon.shape[-1]//4])
    grappa_img = grappa_img/np.max(grappa_img)

    return grappa_img

def sense_reconstruction(ksp,ref_padded,inversed_correction_map = None,thresh=0.003,crop=0):
    mps = mr.app.EspiritCalib(ref_padded,thresh=thresh,crop=crop).run()#(ref_padded,thresh=0.1,crop=0.50).run() #thresh=0.02,crop=0.05 good
    if inversed_correction_map is not None:
        mps = np.multiply(mps,inversed_correction_map[::-1,:])
    sense_img = mr.app.SenseRecon(ksp, mps).run()
    sense_img = complex_image_normalization(sense_img)
    sense_img = sense_img[::-1,:]

    return sense_img

def complex_image_normalization(img):
    #extract phase and magnitude
    phase = np.angle(img)
    magnitude = np.abs(img)
    #normalize magnitude
    magnitude = magnitude - np.min(magnitude)
    magnitude = magnitude/np.max(magnitude)
    #reconstruct complex image
    img = magnitude*np.exp(1j*phase)
    return img


def remove_edges(Zi_body_coils,Zi_surface_coils):
    inter_img_body_coils = abs(Zi_body_coils[:,int(Zi_body_coils.shape[1]//4):-int(Zi_body_coils.shape[1]//4)])#**0.4
    inter_img_body_coils = inter_img_body_coils[:,::-1,...]

    inter_img_surface_coils = abs(Zi_surface_coils[:,int(Zi_surface_coils.shape[1]//4):-int(Zi_surface_coils.shape[1]//4)])#**0.4
    inter_img_surface_coils = inter_img_surface_coils[:,::-1,...]

    return inter_img_body_coils,inter_img_surface_coils

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
