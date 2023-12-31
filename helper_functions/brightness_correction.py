import os
import numpy as np
from helper_functions.preprocess import ifftnd, rms_comb,remove_RO_oversamling
from helper_functions.read_data import readtwix_arry_all
from helper_functions.Interpolation import generate_3D_data, interpolation, quaternion_to_directions
from helper_functions.calculating_correction_map import calculate_correction_map, normalize_image, calculate_correction_map_3D
#from brightness_correction.recon import grappa_reconstruction
from helper_functions.recon import sense_reconstruction, remove_edges, remove_oversampling_phase_direction, rotate_image, pad_ref
from matplotlib import pyplot as plt
import gzip
import pickle

def rotate_images_for_LGE(results, quat, filename):
    if results is None:
        return None

    read_dir , phase_dir, _ = quaternion_to_directions(quat)

    if filename.find('LAX') != -1:
        if read_dir[0] > 0:
            #flip the image
            results = np.flip(results, axis = 2)
        #rotate the image clockwise 90 degree
        results = np.rot90(results,k=-1, axes = (1,2))

    if filename.find('SAX') != -1:
        if read_dir[-1] > 0:
            #flip the image
            results = np.flip(results, axis = 2)
        if phase_dir[0] < 0:
            #flip the image
            results = np.flip(results, axis = 1)
        #rotate the image
        results = np.rot90(results, axes = (1,2))

    if filename.find('2CH') != -1:
        if read_dir[-1] > 0:
            #flip the image
            results = np.flip(results, axis = 2)
        if phase_dir[0] < 0 and phase_dir[1] > 0:
            #flip the image
            results = np.flip(results, axis = 1)
        #rotate the image
        results = np.rot90(results, axes = (1,2))

    return results



def get_dimension_indices(all_dimensions, dimensions_to_find):
    indices = []
    for dim in dimensions_to_find:
        try:
            index = all_dimensions.index(dim)
            indices.append(index)
        except ValueError:
            print(f"Dimension '{dim}' not found, skipping.")
    return indices

def move_dimension_to_front_data(array, all_dimensions, dim_to_move):
    try:
        source = get_dimension_indices(all_dimensions, [dim_to_move])[0]
        return np.moveaxis(array, source, 0)
    except IndexError:
        print(f"Warning: Dimension '{dim_to_move}' not found.")
        return array
    
def move_dimension_to_front_info(all_dimensions, dim_to_move):
    # Check if the dimension exists in the list
    if dim_to_move in all_dimensions:
        all_dimensions.remove(dim_to_move)  # Remove the dimension
        all_dimensions.insert(0, dim_to_move)  # Prepend the dimension
    else:
        print(f"Warning: Dimension '{dim_to_move}' not found.")
    return all_dimensions

def data_reduction(data,data_dimensions, dims_to_keep = ['Sli', 'Lin', 'Cha', 'Col'], dim_to_set_to_zero = ['Phs','Set']):
    ###all_dimensions = ['Ide', 'Idd', 'Idc', 'Idb', 'Ida', 'Seg', 'Set', 'Rep','Phs', 'Eco', 'Par', 'Sli', 'Ave', 'Lin', 'Cha', 'Col']
    for dim in dim_to_set_to_zero:
        if dim in data_dimensions:
            data = move_dimension_to_front_data(data, data_dimensions, dim)
            data_dimensions = move_dimension_to_front_info(data_dimensions, dim)
            data = data[0, ...] # Set the first slice to zero
            #remove the first element in data_dimensions
            data_dimensions = data_dimensions[1:]

    # print("*******************dim_info_org after setting to zeros**********************")
    # dim_info_zip = zip( data_dimensions , data.shape )
    # for i in dim_info_zip:
    #     print(i,end=' ')
    # print("\n")

    shape = data.shape
    slices = [slice(None)] * len(shape)  # By default, keep all of the data for all dimensions

    dims_found = get_dimension_indices(data_dimensions, dims_to_keep)

    for i in range(len(shape)):
        if i not in dims_found:
            # Only slice the dimensions that are not specified in dims_to_keep
            middle_index = shape[i] // 2  # This is the middle index
            slices[i] = slice(middle_index, middle_index + 1)  # Get the middle slice

    data_dimensions = [dim for i, dim in enumerate(data_dimensions) if i in dims_found]

    return np.squeeze(data[tuple(slices)]),data_dimensions  # Squeeze to remove the dimensions that have a length of 1

def target_path_generator(base_dir, input_folder, output_folder, input_subfolders):
    data_path_names = []
    data_path_names_output = []
    input_folder = os.path.join(base_dir, input_folder)
    output_folder = os.path.join(base_dir, output_folder)
    input_subfolders = os.listdir(input_folder) if input_subfolders is None else input_subfolders
    for input_subfolder in input_subfolders:
    # Construct the full path to the directory
        full_dir_name = os.path.join(input_folder , input_subfolder)
        full_dir_name_output = os.path.join(output_folder , input_subfolder)

    # Check if it's a directory
        if os.path.isdir(full_dir_name):
        # Loop over all files in the directory
            for filename in os.listdir(full_dir_name):
            # Check if the file is a .dat file
                if filename.endswith(".dat") or filename.endswith(".gz"):
                    data_path_names.append(os.path.join(full_dir_name, filename))
                    data_path_names_output.append(os.path.join(full_dir_name_output, filename))
    return data_path_names, data_path_names_output

def rawdata_reader(data_path_filename):
    #if data_path_filename ends with .demo.gz, then it is a demo file
    try:
        # start reading the data if these are no reference data or noise data, then the reference data or noise data will be returned as None
        twix, mapped_data,data_org, dim_info_org ,data_ref, dim_info_ref, noise_kspace, dim_info_noise = readtwix_arry_all(data_path_filename=data_path_filename)
        
        # print("\n*******************dim_info_org**********************")
        # dim_info_zip = zip( dim_info_org , data_org.shape )
        # for i in dim_info_zip:
        #     print(i,end=' ')
        # print("\n")

        #this will squeeze the dimensions
        data = data_org.squeeze()
        dim_info_data = dim_info_org.copy()

        #end of reading the data
        
        # start 
        image_3D_body_coils , image_3D_surface_coils = generate_3D_data(mapped_data)
        
        try:
            num_sli = data.shape[dim_info_org.index('Sli')]
        except:
            num_sli = 1
    except:
        # Load the compressed data from the .demo.gz file
        with gzip.open(data_path_filename, 'rb') as f:
            loaded_data = pickle.load(f)

        # Extract the individual datasets from the loaded data
        twix = loaded_data['twix']
        image_3D_body_coils = loaded_data['image_3D_body_coils']
        image_3D_surface_coils = loaded_data['image_3D_surface_coils']
        data = loaded_data['data']
        dim_info_data = loaded_data['dim_info_data']
        data_ref = loaded_data['data_ref']
        dim_info_ref = loaded_data['dim_info_ref']
        num_sli = loaded_data['num_sli']

    
    print('num_sli in the rawdata',num_sli)

    return twix, image_3D_body_coils, image_3D_surface_coils, data, dim_info_data, data_ref, dim_info_ref, num_sli

def low_resolution_img_interpolator(twix, image_3D_body_coils, image_3D_surface_coils, data, dim_info_data, num_sli , auto_rotation = 'Dicom'):

    img_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2))
    sensitivity_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2))
    low_resolution_surface_coil_imgs = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2),dtype=np.complex64)
    img_quats = []

    for n in range(num_sli):
        img_correction_map_all,img_quat,x2d,sensitivity_correction_map_all = calculating_correction_maps(auto_rotation,
                                                                                                                        twix, dim_info_data, 
                                                                                                                        data, image_3D_body_coils, 
                                                                                                                        image_3D_surface_coils, 
                                                                                                                        num_sli, img_correction_map_all,
                                                                                                                        sensitivity_correction_map_all,
                                                                                                                        n)
        low_resolution_surface_coil_imgs[n,...] = x2d
        img_quats.append(img_quat)

    return img_correction_map_all, sensitivity_correction_map_all, low_resolution_surface_coil_imgs, img_quats

from typing import List, Any  # Import necessary types from the typing module

def correction_map_generator(
    twix: List,  
    image_3D_body_coils: np.ndarray,  
    image_3D_surface_coils: np.ndarray,  
    data: np.ndarray,  
    dim_info_data: List[str], 
    num_sli: int,
    auto_rotation: str = 'Dicom',
    lamb: float = 0.5,
    tol: float = 1e-4,
    maxiter: int = 500,
    apply_correction_to_sensitivity_maps: bool = False,
    oversampling_phase_factor: int = 1):
    
    """
    Generates correction maps for MRI images based on low resolution pre-scan.

    This function processes MRI data to generate image correction maps and sensitivity correction maps. 
    It supports handling both single and multiple parallel images. The processing is done slice-by-slice for 
    the specified number of slices.

    Parameters:
    - twix: Overall raw data from the MRI scanner.
    - image_3D_body_coils: Low resolution 3D image data from body coils.
    - image_3D_surface_coils: Low resolution 3D image data from surface coils.
    - data: extracted image data to be processed.
    - dim_info_data: Dimension information for the data.
    - num_sli: Number of slices to process.
    - auto_rotation: Specifies the rotation mode (default is 'Dicom', avaliable values: Dicom, LGE).
    - lamb: Lambda parameter for regularization for finding the correction map(default is 0.5).
    - tol: Tolerance for convergence for finding the correction map (default is 1e-4).
    - maxiter: Maximum number of iterations for finding the correction map (default is 500).
    - apply_correction_to_sensitivity_maps: Flag to apply correction to sensitivity maps while performing sense reconstruction(default is False).
    - oversampling_phase_factor: Factor for phase oversampling, 3 is suggested when you have aliasing in your dataset (default is 1).

    Returns:
    - img_correction_map_all: Array of image correction maps for each slice.
    - sensitivity_correction_map_all: Array of sensitivity correction maps for each slice.
    - low_resolution_surface_coil_imgs: Low-resolution images from surface coils.
    - img_quats: A list containing quaternions for each slice.

    Exceptions:
    - Catches and handles exceptions related to data shape and dimension indexing, defaulting to specific array 
      configurations in case of an error.
    """

    try:
        par_num = data.shape[dim_info_data.index('Par')]
        if par_num > 1:
            img_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2,par_num))
            sensitivity_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2,par_num))
            low_resolution_surface_coil_imgs = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2,par_num),dtype=np.complex64)
        else:
            img_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2))
            sensitivity_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2))
            low_resolution_surface_coil_imgs = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2),dtype=np.complex64)
    except:
        img_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2))
        sensitivity_correction_map_all = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2))
        low_resolution_surface_coil_imgs = np.zeros((num_sli,data.shape[dim_info_data.index('Lin')],data.shape[dim_info_data.index('Col')]//2),dtype=np.complex64)


    img_quats = []
    correction_map3D, correction_map3D_sense = None , None

    for n in range(num_sli):
        img_correction_map_all,img_quat,low_resolution_surface_coil_imgs,sensitivity_correction_map_all,correction_map3D, correction_map3D_sense = calculating_correction_maps(auto_rotation,
                                                                                                                        twix, dim_info_data, 
                                                                                                                        data, image_3D_body_coils, 
                                                                                                                        image_3D_surface_coils, 
                                                                                                                        num_sli, img_correction_map_all,
                                                                                                                        sensitivity_correction_map_all,
                                                                                                                        low_resolution_surface_coil_imgs,
                                                                                                                        n, lamb = lamb, tol = tol, maxiter=maxiter,
                                                                                                                        apply_correction_to_sensitivity_maps = apply_correction_to_sensitivity_maps,
                                                                                                                        correction_map3D = correction_map3D, correction_map3D_sense = correction_map3D_sense,
                                                                                                                        oversampling_phase_factor = oversampling_phase_factor)

        img_quats.append(img_quat)

    return img_correction_map_all, sensitivity_correction_map_all, low_resolution_surface_coil_imgs, img_quats

def auto_image_rotation(sense_recon_results, img_quats, auto_rotation = 'Dicom', img_correction_map = None, sens_correction_map = None,filename = None):

    if auto_rotation not in ['Dicom','LGE']:
        raise ValueError("auto_rotation should be either 'Dicom' or 'LGE'")
    
    if auto_rotation == 'Dicom':
        #num_sli = sense_recon_results.shape[0]
        #for n in range(num_sli):
        #    sense_recon_results_rotated = sense_img_rotation(sense_recon_results_rotated, sense_recon_results[n,...], img_quats[n], num_sli, n, auto_rotation = auto_rotation)
        #return sense_recon_results_rotated, img_correction_map, sens_correction_map
        return sense_recon_results, img_correction_map, sens_correction_map
    
    if auto_rotation == 'LGE':
        sense_recon_results_rotated = np.zeros((sense_recon_results.shape),dtype=np.complex64)
        sense_recon_results_rotated = rotate_images_for_LGE(sense_recon_results, img_quats[0], filename)
        img_correction_map = rotate_images_for_LGE(img_correction_map, img_quats[0], filename)
        sens_correction_map = rotate_images_for_LGE(sens_correction_map, img_quats[0], filename)
        return sense_recon_results_rotated, img_correction_map, sens_correction_map
                

    


def sense_img_rotation(sense_recon_results, sense_reconstructed_img, img_quat, num_sli, n, auto_rotation = 'Dicom'):
    if auto_rotation == 'Dicom':
        sense_reconstructed_img = np.rot90(sense_reconstructed_img,-1)
        sense_reconstructed_img = rotate_image(sense_reconstructed_img,img_quat)

    print("sense_recon_results",sense_recon_results.shape)
    print("sense_reconstructed_img",sense_reconstructed_img.shape)
    try:
        sense_recon_results[n,...] = sense_reconstructed_img
    except:
        sense_recon_results = np.zeros((num_sli,sense_reconstructed_img.shape[-2],sense_reconstructed_img.shape[-1]),dtype=np.complex64)
        sense_recon_results[n,...] = sense_reconstructed_img              

    return sense_recon_results

def save_sense_recon_results(full_dir_name_output,sense_recon_results, img_correction_map, sens_correction_map, quat, apply_correction_during_sense_recon):
    full_dir_name_output = full_dir_name_output[:-4]
    # Save the results
    if apply_correction_during_sense_recon:
        corrected_results_filename = full_dir_name_output + ".corrected_sense_results.npy"
    else:
        uncorrected_results_filename = full_dir_name_output + ".uncorrected_results.npy"

    correction_map_all_filename = full_dir_name_output + ".image_correction_map.npy"
    inversed_correction_map_all_filename = full_dir_name_output + ".sensitivity_correction_map.npy"
    quat_filename = full_dir_name_output + ".quat.npy"
    # Save reconstruction_results and correction_map_all to their respective files
    os.makedirs(os.path.dirname(full_dir_name_output), exist_ok=True)

    if apply_correction_during_sense_recon:
        np.save(corrected_results_filename, sense_recon_results)
    else:
        np.save(uncorrected_results_filename, sense_recon_results)

    if np.sum(img_correction_map) != 0 and img_correction_map is not None and np.any(~np.isnan(img_correction_map)):
        np.save(correction_map_all_filename, img_correction_map)
    if np.sum(sens_correction_map) != 0 and sens_correction_map is not None and np.any(~np.isnan(sens_correction_map)):
        np.save(inversed_correction_map_all_filename, sens_correction_map)
    
    #save quat and slc_dir for future debugging
    try:
        np.save(quat_filename, quat[0])   
    except:
        ...
    


def brightness_correction_map_generator(data_path, filename_matched, auto_rotation = 'Dicom', apply_correction_during_sense_recon = False, CustomProcedure = None):
    
    #twix, data_org, dim_info_org,data_ref, dim_info_ref, noise_kspace, dim_info_noise = readtwix_arry_all(data_path, filename_matched)
    twix, mapped_data,data_org, dim_info_org ,data_ref, dim_info_ref, noise_kspace, dim_info_noise = readtwix_arry_all(data_path = data_path, filename = filename_matched)
    print("dim_info_org\n",dim_info_org,'\n',data_org.shape)

    data = data_org.squeeze()

        #all possilbe dim_info_org                                              *       *
        #"Ide", "Idd", "Idc", "Idb", "Ida", "Seg", "Set", "Rep","Phs", "Eco", "Par", "Sli", "Ave", "Lin", "Cha", "Col"
    try:
        try:
            dim_info_org.index('Set')
            print("multiple sets")
            C = None
        except:
            dim_info_org.index('Sli')
            print("multiple slices")
            C = None            
    except:
        image = ifftnd(data, [dim_info_org.index('Lin'),dim_info_org.index('Col')])
        image = rms_comb(image)
        print("single sets")
        image = image[0,1,14,...]
        image = abs(image[:,int(image.shape[1]/4):-int(image.shape[1]/4)])
        C = image[::-1,:]


    image_3D_body_coils , image_3D_surface_coils = generate_3D_data(mapped_data)
    #print("\n!!!!!!!!!!!!!"+str(image_3D_body_coils.shape)+"   "+str(image_3D_surface_coils.shape)+"!!!!!!!!!!!!!!\n")
    #                   !!!!!!!!!!!!!(128, 32, 32, 2)                      (128, 32, 32, 30)!!!!!!!!!!!!!!

    try:
        num_sli = data.shape[dim_info_org.index('Sli')]
        
    except:
        num_sli = 1
    #print('num_sli',num_sli)

    correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Lin')],data.shape[dim_info_org.index('Col')]//2))
    inversed_correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Lin')],data.shape[dim_info_org.index('Col')]//2))
    #grappa_results_raw = np.zeros((num_sli,data.shape[dim_info_org.index('Ave')],data.shape[dim_info_org.index('Lin')],data.shape[dim_info_org.index('Col')]//2))
    recon_results     = np.zeros((num_sli,data.shape[dim_info_org.index('Lin')],data.shape[dim_info_org.index('Col')]//2),dtype=np.complex64)



    # print("*******************dim_info_org before reducing dimentions**********************")
    # dim_info_zip = zip( dim_info_org , data.shape )
    # for i in dim_info_zip:
    #     print(i,end=' ')
    # print("\n")


    data,dim_info_org = data_reduction(data,data_dimensions = dim_info_org, dims_to_keep = ['Sli', 'Lin', 'Cha', 'Col'])

    # print("*******************dim_info_org after reducing dimensions**********************")
    # dim_info_zip = zip( dim_info_org , data.shape )
    # for i in dim_info_zip:
    #     print(i,end=' ')
    # print("\n")

    for n in range(num_sli):

        try:
            if 'image' not in locals():

                ksp,ref_padded = pad_ref(data,data_ref,n,dim_info_ref = dim_info_ref,dim_info_org=dim_info_org)
                ksp = remove_RO_oversamling(ksp,axis_RO=2)
                ref_padded = remove_RO_oversamling(ref_padded,axis_RO=2)

                ###interpolation starts here
                correction_map_all, ksp, ref_padded, img_quat, x2d, inversed_correction_map_all = calculating_correction_maps(auto_rotation, 
                                                                                                                                    CustomProcedure, 
                                                                                                                                    twix, dim_info_org, 
                                                                                                                                    data, image_3D_body_coils, 
                                                                                                                                    image_3D_surface_coils, 
                                                                                                                                    num_sli, correction_map_all,
                                                                                                                                    inversed_correction_map_all,
                                                                                                                                    n)#,ksp,ref_padded,noise_kspace, dim_info_noise)
                ###interpolation ends here with correction map and inversed correction map gnerated

                if ref_padded is None:
                    #we don't need to do reconstruction
                    C = ifftnd(ksp, [-1,-2])
                    C = rms_comb(C, axis=0)
                
                else: #doing the reconstruction
                    ##grappa reconstruction #if you want to use grappa reconstruction
                    # uncomment the following line and the parts about grappa reconstruction
                    #C = grappa_reconstruction(ksp,ref_padded)

                    #sense reconstruction
                    print("apply_correction_during_sense_recon: ",apply_correction_during_sense_recon)
                    if apply_correction_during_sense_recon:
                        C = sense_reconstruction(ksp,ref_padded,inversed_correction_map_all[n,...])
                    else:
                        C = sense_reconstruction(ksp,ref_padded)

                if auto_rotation == 'Dicom':
                    C = np.rot90(C,-1)
                    C = rotate_image(C,img_quat)
                try:
                    recon_results[n,...] = C
                except:
                    recon_results = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]),dtype=np.complex64)
                    recon_results[n,...] = C
        except Exception as e:
            print(e)
            correction_map_all, ksp, ref_padded, img_quat, x2d, inversed_correction_map_all = calculating_correction_maps(auto_rotation, 
                                                                                                                                      CustomProcedure, 
                                                                                                                                      twix, dim_info_org, 
                                                                                                                                      data, image_3D_body_coils, 
                                                                                                                                      image_3D_surface_coils, 
                                                                                                                                      num_sli, correction_map_all,
                                                                                                                                      inversed_correction_map_all,
                                                                                                                                      n)
            #print the warning with color
            print("\033[91m" + "Can't use the default sense reconstruction method! display low resolution interpolated image instead!" + "\033[0m")
            recon_results[n,...] = x2d

    #return A,B,grappa_results,correction_map_all
    return recon_results,correction_map_all,img_quat, inversed_correction_map_all, apply_correction_during_sense_recon

def single_img_interpolation(twix, image_3D_body_coils, image_3D_surface_coils, 
                                num_sli, n,
                                ):
    

    Zi_body_coils, Zi_surface_coils, img_quat , _= interpolation(twix, image_3D_body_coils, image_3D_surface_coils, num_sli, n)
    inter_img_body_coils, inter_img_surface_coils = remove_edges(Zi_body_coils,Zi_surface_coils)
    inter_img_body_coils = inter_img_body_coils.transpose([2,0,1])
    inter_img_surface_coils = inter_img_surface_coils.transpose([2,0,1])


    return inter_img_body_coils, inter_img_surface_coils, img_quat

def single_correction_map_generator(auto_rotation, img_quat, dim_info_org, 
                                data, inter_img_body_coils, inter_img_surface_coils, 
                                num_sli, correction_map_all,inversed_correction_map_all, n,
                                ):
    

    inter_img_body_coils = rms_comb(inter_img_body_coils,0)
    inter_img_surface_coils = rms_comb(inter_img_surface_coils,0)


    x2d = normalize_image(inter_img_surface_coils)#,scanned_img)
    x3d = normalize_image(inter_img_body_coils)

    *_ ,correction_map = calculate_correction_map(A=x2d,B=x3d,lamb=1e-1)
    *_ ,inversed_correction_map = calculate_correction_map(A=x3d,B=x2d,lamb=1e-1)

    correction_map_all, inversed_correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                                                        correction_map_all, inversed_correction_map_all, n, 
                                                                                        img_quat, correction_map, inversed_correction_map)
                                                                        
    return correction_map_all,img_quat,x2d,inversed_correction_map_all

import cv2


# Normalize and convert the images to uint8
def normalize_convert_uint8(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    uint8_image = (normalized_image * 255).astype('uint8')
    return uint8_image

def find_and_combine_contours(image_3D_body_coils, image_3D_surface_coils):

    # Process the image and find edges
    def find_edges(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary_image, 50, 150)
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        return dilated_edges

    # Draw contours on a blank image
    def draw_contours_on_blank(image, contours):
        blank_image = np.zeros_like(image)
        cv2.drawContours(blank_image, contours, -1, (255, 255, 255), 1)
        return blank_image

    # Normalize and convert both images to uint8
    image_3D_body_coils_uint8 = normalize_convert_uint8(image_3D_body_coils)
    image_3D_surface_coils_uint8 = normalize_convert_uint8(image_3D_surface_coils)

    # Find edges for both images
    edges_body_coils = find_edges(image_3D_body_coils_uint8)
    edges_surface_coils = find_edges(image_3D_surface_coils_uint8)

    # Combine the edge images
    combined_edges = (edges_body_coils // 2 + edges_surface_coils // 2).astype(np.uint8)

    # Find the combined contours
    combined_contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the combined contours on a blank image
    combined_contour_image = draw_contours_on_blank(image_3D_body_coils_uint8, combined_contours)

    return combined_contour_image, combined_contours

def create_energy_based_mask(image, contours):
    # Initialize the mask with zeros
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        # Check that the contour is not empty
        if contour.size > 0:
            # Create a binary mask where the area inside the contour is 1, and outside is 0
            contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)
            
            # Calculate the average energy inside and outside the contour
            inside_average = np.mean(image[contour_mask == 1])
            outside_average = np.mean(image[contour_mask == 0])
            
            # Assign the area with the lowest average energy with zero, the rest with one
            if inside_average < outside_average:
                mask[contour_mask == 1] = 0
            else:
                mask[contour_mask == 1] = 1
    
    return mask


def apply_correction_mask(image_3D_body_coils_3D, image_3D_surface_coils_3D, correction_map3D, thredshold = 0.15):
    """
    Create masks from image_3D_body_coils_3D and image_3D_surface_coils_3D based on 0.1 percentile threshold.
    Combine the masks and apply the combined mask to correction_map3D.
    
    Parameters:
    - image_3D_body_coils_3D: numpy array
    - image_3D_surface_coils_3D: numpy array
    - correction_map3D: numpy array
    
    Returns:
    - corrected_map: numpy array
    """

    #save the images into npy files
    #np.save("image_3D_body_coils_3D.npy",image_3D_body_coils_3D)
    #np.save("image_3D_surface_coils_3D.npy",image_3D_surface_coils_3D)

    # body_coil_max = np.max(image_3D_body_coils_3D)
    # body_coil_min = np.min(image_3D_body_coils_3D)
    # sufface_coil_max = np.max(image_3D_surface_coils_3D)
    # sufface_coil_min = np.min(image_3D_surface_coils_3D)


    # threshold_body = (body_coil_max - body_coil_min) * thredshold + body_coil_min
    # threshold_surface = (sufface_coil_max - sufface_coil_min) * thredshold + sufface_coil_min

    # # Create masks based on the thresholds
    # mask_body = (image_3D_body_coils_3D > threshold_body).astype(int)
    # mask_surface = (image_3D_surface_coils_3D > threshold_surface).astype(int)

    # print(threshold_body,threshold_surface)


    # #plot masks 
    # plt.imshow(mask_body)#[:,16,:])
    # plt.show()
    # plt.imshow(mask_surface)#[:,16,:])
    # plt.show()

    combined_contour_image, combined_contours = find_and_combine_contours(image_3D_body_coils_3D, image_3D_surface_coils_3D)
    # plt.imshow(combined_contour_image, cmap='gray')
    # plt.title('Combined Contours')
    # plt.axis('off')
    # plt.show()


    mask = create_energy_based_mask(image_3D_body_coils_3D+image_3D_surface_coils_3D, combined_contours)
    # plt.imshow(mask, cmap='gray')
    # plt.title('Energy-Based Mask')
    # plt.axis('off')
    # plt.show()

    combined_contour_image = combined_contour_image.astype(np.float64)


    #plot image_3D_body_coils_3D and image_3D_surface_coils_3D
    img_temp = image_3D_body_coils_3D+(combined_contour_image/255)*np.max(image_3D_body_coils_3D)
    # plt.imshow(img_temp)#[:,16,:])
    # plt.colorbar()
    # plt.show()

    img_temp = image_3D_surface_coils_3D+(combined_contour_image/255)*np.max(image_3D_surface_coils_3D)
    # plt.imshow(img_temp)#[:,16,:])
    # plt.colorbar()
    # plt.show()


    # Apply combined mask to correction_map3D
    corrected_map = correction_map3D * mask
    
    return corrected_map

def calculating_correction_maps(auto_rotation, twix, dim_info_org, 
                                data, image_3D_body_coils, image_3D_surface_coils, 
                                num_sli, correction_map_all,inversed_correction_map_all, low_resolution_surface_coil_imgs,n, lamb = 1e-3,tol = 1e-4, maxiter=500, 
                                apply_correction_to_sensitivity_maps = False, correction_map3D = None , correction_map3D_sense = None,oversampling_phase_factor = 1
                                ):

    if correction_map3D is None and correction_map3D_sense is None:
        image_3D_body_coils_3D = rms_comb(image_3D_body_coils.copy(),-1)
        image_3D_surface_coils_3D = rms_comb(image_3D_surface_coils.copy(),-1)

        print("image_3D_body_coils_3D",image_3D_body_coils_3D.shape)
        print("image_3D_surface_coils_3D",image_3D_surface_coils_3D.shape)

        if apply_correction_to_sensitivity_maps == False:
            correction_map3D = calculate_correction_map_3D(image_3D_surface_coils_3D,image_3D_body_coils_3D,lamb=lamb,tol=tol, maxiter=maxiter, sensitivity_correction_maps=False, debug = False)
            correction_map3D_sense = None
            #correction_map3D = apply_correction_mask(image_3D_body_coils_3D, image_3D_surface_coils_3D, correction_map3D)
        else:
            correction_map3D = None
            correction_map3D_sense = calculate_correction_map_3D(image_3D_surface_coils_3D,image_3D_body_coils_3D,lamb=lamb,tol=tol, maxiter=maxiter, sensitivity_correction_maps=True, debug = False)
            #correction_map3D_sense = apply_correction_mask(image_3D_body_coils_3D, image_3D_surface_coils_3D, correction_map3D_sense)
        
    
    else:
        ...


    interpolated_data, img_quat, _ = interpolation(twix, data, dim_info_org,num_sli, n, [image_3D_body_coils, image_3D_surface_coils, correction_map3D, correction_map3D_sense] , oversampling_phase_factor = oversampling_phase_factor)
    inter_img_body_coils, inter_img_surface_coils, correction_map_from_3D, correction_map_from_3D_sense = interpolated_data
    
    inter_img_surface_coils = rms_comb(inter_img_surface_coils.copy(),-1)
    inter_img_body_coils = rms_comb(inter_img_body_coils.copy(),-1)



    # print("dim_info_org",dim_info_org)
    # try:
    #     print("inter_img_surface_coils.shape",inter_img_surface_coils.shape)
    # except:
    #     ...
    # try:
    #     print("inter_img_body_coils.shape",inter_img_body_coils.shape)
    # except:
    #     ...
    # try:
    #     print("correction_map_from_3D.shape",correction_map_from_3D.shape)
    # except:
    #     ...
    # try:
    #     print("correction_map_from_3D_sense.shape",correction_map_from_3D_sense.shape)
    # except:
    #     ...

    # try:
    #     print("correction_map_all.shape",correction_map_all.shape)
    # except:
    #     ...


    #for 4d:
    # inter_img_surface_coils.shape (96, 192, 72)
    # inter_img_body_coils.shape (96, 192, 72)
    # correction_map_from_3D.shape (96, 192, 72)


    #for 3d:
    # inter_img_surface_coils.shape (150, 512)
    # inter_img_body_coils.shape (150, 512)
    # correction_map_from_3D.shape (150, 512)
    # correction_map_all.shape (1, 150, 256)

    try:
        par_num = data.shape[dim_info_org.index('Par')]
        print("par_num",par_num)
        if par_num > 1:
            x2d = np.zeros((data.shape[dim_info_org.index('Lin')],data.shape[dim_info_org.index('Col')]//2,par_num))
            for par_idx in range(par_num):
                if apply_correction_to_sensitivity_maps == False:
                    correction_map_all[...,par_idx], inversed_correction_map_all[...,par_idx], low_resolution_surface_coil_imgs[...,par_idx] = post_processing(auto_rotation, dim_info_org, data, num_sli, n, 
                                                                                    apply_correction_to_sensitivity_maps, oversampling_phase_factor, img_quat, 
                                                                                    inter_img_body_coils[...,par_idx], inter_img_surface_coils[...,par_idx], correction_map_from_3D[...,par_idx], None,
                                                                                    correction_map_all[...,par_idx], None,low_resolution_surface_coil_imgs[...,par_idx])
                else:
                    correction_map_all[...,par_idx], inversed_correction_map_all[...,par_idx], low_resolution_surface_coil_imgs[...,par_idx] = post_processing(auto_rotation, dim_info_org, data, num_sli, n, 
                                                                                    apply_correction_to_sensitivity_maps, oversampling_phase_factor, img_quat, 
                                                                                    inter_img_body_coils[...,par_idx], inter_img_surface_coils[...,par_idx], None, correction_map_from_3D_sense[...,par_idx],
                                                                                    None, inversed_correction_map_all[...,par_idx],low_resolution_surface_coil_imgs[...,par_idx])
        else:
            correction_map_all, inversed_correction_map_all, low_resolution_surface_coil_imgs = post_processing(auto_rotation, dim_info_org, data, num_sli, n, 
                                                                                   apply_correction_to_sensitivity_maps, oversampling_phase_factor, img_quat, 
                                                                                   inter_img_body_coils, inter_img_surface_coils, correction_map_from_3D, correction_map_from_3D_sense,
                                                                                   correction_map_all, inversed_correction_map_all,low_resolution_surface_coil_imgs)
    except:
        correction_map_all, inversed_correction_map_all, low_resolution_surface_coil_imgs = post_processing(auto_rotation, dim_info_org, data, num_sli, n, 
                                                                               apply_correction_to_sensitivity_maps, oversampling_phase_factor, img_quat, 
                                                                               inter_img_body_coils, inter_img_surface_coils, correction_map_from_3D, correction_map_from_3D_sense,
                                                                               correction_map_all, inversed_correction_map_all,low_resolution_surface_coil_imgs)

    
                                                                        
    return correction_map_all,img_quat,low_resolution_surface_coil_imgs,inversed_correction_map_all, correction_map3D, correction_map3D_sense

def post_processing(auto_rotation, dim_info_org, data, num_sli, n, apply_correction_to_sensitivity_maps, 
                    oversampling_phase_factor, img_quat, inter_img_body_coils, inter_img_surface_coils, 
                    correction_map_from_3D, correction_map_from_3D_sense,correction_map_all, inversed_correction_map_all,low_resolution_surface_coil_imgs):
    
    if oversampling_phase_factor != 1:
        if apply_correction_to_sensitivity_maps == False:
            correction_map_from_3D = apply_correction_mask(inter_img_surface_coils, inter_img_body_coils, correction_map_from_3D)
        else:
            correction_map_from_3D_sense = apply_correction_mask(inter_img_surface_coils, inter_img_body_coils, correction_map_from_3D_sense)


    #remove extra area for the image
    #print("before removing: correction_map_from_3D", correction_map_from_3D.shape)
    # correction_map_from_3D = remove_edges(correction_map_from_3D)
    # correction_map_from_3D_sense = remove_edges(correction_map_from_3D_sense)
    # correction_map_from_3D = np.nan_to_num(correction_map_from_3D, nan=0)
    # correction_map_from_3D_sense = np.nan_to_num(correction_map_from_3D_sense, nan=0)
    #remove over sampled area for the image
    correction_map_from_3D = remove_edges(correction_map_from_3D)
    correction_map_from_3D_sense = remove_edges(correction_map_from_3D_sense)

    correction_map_from_3D = remove_oversampling_phase_direction(correction_map_from_3D, oversampling_phase_factor = oversampling_phase_factor)
    correction_map_from_3D_sense = remove_oversampling_phase_direction(correction_map_from_3D_sense, oversampling_phase_factor = oversampling_phase_factor)

    ############old 2d method#############
    #fill nan values with 0 for inter_img_surface_coils
    #inter_img_body_coils = np.nan_to_num(inter_img_body_coils)
    #inter_img_body_coils = remove_edges(inter_img_body_coils)
    #inter_img_surface_coils = np.nan_to_num(inter_img_surface_coils, nan=0)
    inter_img_surface_coils = remove_edges(inter_img_surface_coils)
    inter_img_surface_coils = remove_oversampling_phase_direction(inter_img_surface_coils, oversampling_phase_factor = oversampling_phase_factor)


    #inter_img_body_coils = inter_img_body_coils.transpose([2,0,1]) 
    #inter_img_surface_coils = inter_img_surface_coils.transpose([2,0,1])

    #inter_img_body_coils = rms_comb(inter_img_body_coils,0)
    #inter_img_surface_coils = rms_comb(inter_img_surface_coils,0)

    x2d = normalize_image(inter_img_surface_coils)
    #x2d = remove_oversampling_phase_direction(x2d, oversampling_phase_factor = oversampling_phase_factor)
    #x3d = normalize_image(inter_img_body_coils)
    # *_ ,correction_map = calculate_correction_map(A=x2d,B=x3d,lamb=lamb)
    # *_ ,inversed_correction_map = calculate_correction_map(A=x3d,B=x2d,lamb=lamb)
    ######################################


    #replace and output correction_map with correction_map_from_3D
    correction_map = correction_map_from_3D
    inversed_correction_map = correction_map_from_3D_sense

    # print("x2d.shape"   ,x2d.shape)
    # print("correction_map_all.shape", correction_map_all.shape)
    # print("correction_map.shape", correction_map.shape)

    #4d
    # x2d.shape (96, 96)
    # correction_map_all.shape (1, 96, 96)
    # correction_map.shape (96, 96)


    if apply_correction_to_sensitivity_maps == False:
        correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                     correction_map_all, n, img_quat, correction_map)
        inversed_correction_map = None
    else:
        correction_map_all = None
        inversed_correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                    inversed_correction_map_all, n, img_quat, inversed_correction_map)
        

    low_resolution_surface_coil_imgs = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                low_resolution_surface_coil_imgs, n, img_quat, x2d)        
    
                                                    
    return correction_map_all,inversed_correction_map_all,low_resolution_surface_coil_imgs

# def calculating_correction_maps(auto_rotation, twix, dim_info_org, 
#                                 data, image_3D_body_coils, image_3D_surface_coils, 
#                                 num_sli, correction_map_all,inversed_correction_map_all, n, lamb = 1e-3,tol = 1e-4, maxiter=500, 
#                                 apply_correction_to_sensitivity_maps = False, correction_map3D = None , correction_map3D_sense = None,oversampling_phase_factor = 1
#                                 ):

#     if correction_map3D is None and correction_map3D_sense is None:
#         image_3D_body_coils_3D = rms_comb(image_3D_body_coils.copy(),-1)
#         image_3D_surface_coils_3D = rms_comb(image_3D_surface_coils.copy(),-1)

#         print("image_3D_body_coils_3D",image_3D_body_coils_3D.shape)
#         print("image_3D_surface_coils_3D",image_3D_surface_coils_3D.shape)

#         if apply_correction_to_sensitivity_maps == False:
#             correction_map3D = calculate_correction_map_3D(image_3D_surface_coils_3D,image_3D_body_coils_3D,lamb=lamb,tol=tol, maxiter=maxiter, sensitivity_correction_maps=False, debug = True)
#             correction_map3D_sense = None
#             #correction_map3D = apply_correction_mask(image_3D_body_coils_3D, image_3D_surface_coils_3D, correction_map3D)
#         else:
#             correction_map3D = None
#             correction_map3D_sense = calculate_correction_map_3D(image_3D_surface_coils_3D,image_3D_body_coils_3D,lamb=lamb,tol=tol, maxiter=maxiter, sensitivity_correction_maps=True, debug = True)
#             #correction_map3D_sense = apply_correction_mask(image_3D_body_coils_3D, image_3D_surface_coils_3D, correction_map3D_sense)
        
    
#     else:
#         ...


#     interpolated_data, img_quat, _ = interpolation(twix, data, dim_info_org,num_sli, n, [image_3D_body_coils, image_3D_surface_coils, correction_map3D, correction_map3D_sense] , oversampling_phase_factor = oversampling_phase_factor)
#     inter_img_body_coils, inter_img_surface_coils, correction_map_from_3D, correction_map_from_3D_sense = interpolated_data
    
#     inter_img_surface_coils = rms_comb(inter_img_surface_coils.copy(),-1)
#     inter_img_body_coils = rms_comb(inter_img_body_coils.copy(),-1)
#     if oversampling_phase_factor != 1:
#         if apply_correction_to_sensitivity_maps == False:
#             correction_map_from_3D = apply_correction_mask(inter_img_surface_coils, inter_img_body_coils, correction_map_from_3D)
#         else:
#             correction_map_from_3D_sense = apply_correction_mask(inter_img_surface_coils, inter_img_body_coils, correction_map_from_3D_sense)
#     print("dim_info_org",dim_info_org)
#     try:
#         print("inter_img_surface_coils.shape",inter_img_surface_coils.shape)
#     except:
#         ...
#     try:
#         print("inter_img_body_coils.shape",inter_img_body_coils.shape)
#     except:
#         ...
#     try:
#         print("correction_map_from_3D.shape",correction_map_from_3D.shape)
#     except:
#         ...
#     try:
#         print("correction_map_from_3D_sense.shape",correction_map_from_3D_sense.shape)
#     except:
#         ...

#     #remove extra area for the image
#     #print("before removing: correction_map_from_3D", correction_map_from_3D.shape)
#     # correction_map_from_3D = remove_edges(correction_map_from_3D)
#     # correction_map_from_3D_sense = remove_edges(correction_map_from_3D_sense)
#     # correction_map_from_3D = np.nan_to_num(correction_map_from_3D, nan=0)
#     # correction_map_from_3D_sense = np.nan_to_num(correction_map_from_3D_sense, nan=0)
#     #remove over sampled area for the image
#     correction_map_from_3D = remove_edges(correction_map_from_3D)
#     correction_map_from_3D_sense = remove_edges(correction_map_from_3D_sense)

#     correction_map_from_3D = remove_oversampling_phase_direction(correction_map_from_3D, oversampling_phase_factor = oversampling_phase_factor)
#     correction_map_from_3D_sense = remove_oversampling_phase_direction(correction_map_from_3D_sense, oversampling_phase_factor = oversampling_phase_factor)

#     ############old 2d method#############
#     #fill nan values with 0 for inter_img_surface_coils
#     #inter_img_body_coils = np.nan_to_num(inter_img_body_coils)
#     #inter_img_body_coils = remove_edges(inter_img_body_coils)
#     #inter_img_surface_coils = np.nan_to_num(inter_img_surface_coils, nan=0)
#     inter_img_surface_coils = remove_edges(inter_img_surface_coils)
#     inter_img_surface_coils = remove_oversampling_phase_direction(inter_img_surface_coils, oversampling_phase_factor = oversampling_phase_factor)


#     #inter_img_body_coils = inter_img_body_coils.transpose([2,0,1]) 
#     #inter_img_surface_coils = inter_img_surface_coils.transpose([2,0,1])

#     #inter_img_body_coils = rms_comb(inter_img_body_coils,0)
#     #inter_img_surface_coils = rms_comb(inter_img_surface_coils,0)

#     x2d = normalize_image(inter_img_surface_coils)
#     #x2d = remove_oversampling_phase_direction(x2d, oversampling_phase_factor = oversampling_phase_factor)
#     #x3d = normalize_image(inter_img_body_coils)
#     # *_ ,correction_map = calculate_correction_map(A=x2d,B=x3d,lamb=lamb)
#     # *_ ,inversed_correction_map = calculate_correction_map(A=x3d,B=x2d,lamb=lamb)
#     ######################################


#     #replace and output correction_map with correction_map_from_3D
#     correction_map = correction_map_from_3D
#     inversed_correction_map = correction_map_from_3D_sense


#     if apply_correction_to_sensitivity_maps == False:
#         correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
#                                                      correction_map_all, n, img_quat, correction_map)
#         inversed_correction_map = None
#     else:
#         correction_map_all = None
#         inversed_correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
#                                                     inversed_correction_map_all, n, img_quat, inversed_correction_map)


                                                                        
#     return correction_map_all,img_quat,x2d,inversed_correction_map_all, correction_map3D, correction_map3D_sense

# def calculating_correction_maps(auto_rotation, CustomProcedure, twix, dim_info_org, 
#                                 data, image_3D_body_coils, image_3D_surface_coils, 
#                                 num_sli, correction_map_all,inversed_correction_map_all, n,
#                                 ksp = None , ref_padded = None , noise_kspace = None , dim_info_noise = None ):
    
#     Zi_body_coils, Zi_surface_coils, img_quat , normal= interpolation(twix, image_3D_body_coils, image_3D_surface_coils, num_sli, n)
#     inter_img_body_coils, inter_img_surface_coils = remove_edges(Zi_body_coils,Zi_surface_coils)
#     inter_img_body_coils = inter_img_body_coils.transpose([2,0,1])
#     inter_img_surface_coils = inter_img_surface_coils.transpose([2,0,1])

#     #see if these is a function called CustomProcedure else use defaultProcedure
#     try:
#         inter_img_body_coils, inter_img_surface_coils, ksp, ref_padded = CustomProcedure(inter_img_body_coils,
#                                                                                                          inter_img_surface_coils, 
#                                                                                                          ksp, ref_padded, noise_kspace, dim_info_noise)
#         #print with green color
#         print("\033[92m" + "custom procedure found and used!" + "\033[0m")
#     except Exception as e:
#         if CustomProcedure != None:
#             #print with red color
#             print("\033[91m" + "custom procedure found, but there is something wrong with it. Use default procedure!" + "\033[0m")
#             print(e)
#         else:
#             #print with yellow color
#             print("\033[93m" + "no custom procedure found, use default procedure!" + "\033[0m")
#         inter_img_body_coils, inter_img_surface_coils, ksp, ref_padded = defaultProcedure(inter_img_body_coils,
#                                                                                                           inter_img_surface_coils, 
#                                                                                                           ksp, ref_padded, noise_kspace, dim_info_noise)
        

#     x2d, x3d = normalize_images(inter_img_surface_coils,inter_img_body_coils)#,scanned_img)


#     *_ ,correction_map = calculate_correction_map(A=x2d,B=x3d,lamb=1e-1)
#     *_ ,inversed_correction_map = calculate_correction_map(A=x3d,B=x2d,lamb=1e-1)

#     correction_map_all, inversed_correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
#                                                                                         correction_map_all, inversed_correction_map_all, n, 
#                                                                                         img_quat, correction_map, inversed_correction_map)
                                                                        
#     return correction_map_all,ksp,ref_padded,img_quat,normal,x2d,inversed_correction_map_all

def correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, correction_map_all
                            , n, img_quat, correction_map):
    
    #throw an error if auto_rotation is nither 'Dicom' nor 'LGE'
    if auto_rotation not in ['Dicom','LGE']:
        raise ValueError("auto_rotation should be either 'Dicom' or 'LGE'")


    correction_map = np.abs(correction_map)
    if auto_rotation == 'Dicom':
        correction_map = np.rot90(correction_map,-1)

    try:
        if auto_rotation == 'Dicom':
            correction_map_all[n,...] = rotate_image(correction_map,img_quat)
        else:
            correction_map_all[n,...] = correction_map
    except:
        if auto_rotation == 'Dicom':
            correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
            correction_map_all[n,...] = rotate_image(correction_map,img_quat)
        else:
            correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
            correction_map_all[n,...] = correction_map
    return correction_map_all

def sense_input_rotation(auto_rotation, img_quat, sense_data):

    if auto_rotation not in ['Dicom','LGE']:
        raise ValueError("auto_rotation should be either 'Dicom' or 'LGE'")

    if auto_rotation != 'Dicom':
        return sense_data[...,::-1,:]
    
    else:
        sense_data = np.rot90(sense_data,-1,[-2,-1])
        sense_data = sense_data[:,...,::-1]
        sense_data_rotated = np.zeros((sense_data.shape),dtype=np.complex64)
        for ch_idx in range(sense_data.shape[0]):
            try:
                sense_data_rotated[ch_idx,...] = rotate_image(sense_data[ch_idx,...],img_quat)
            except:
                sense_data_rotated = np.zeros((sense_data.shape[0],sense_data.shape[-1],sense_data.shape[-2]),dtype=np.complex64)
                sense_data_rotated[ch_idx,...] = rotate_image(sense_data[ch_idx,...],img_quat)

        return sense_data_rotated

# def sense_input_rotation(auto_rotation, img_quat, sense_data):
#     if auto_rotation != 'Dicom':
#         return sense_data
    
#     else:
#         sense_data = np.rot90(sense_data,-1,[-2,-1])
#         sense_data_rotated = np.zeros((sense_data.shape),dtype=np.complex64)
#         different_shape_flag = False
#         for ch_idx in range(sense_data.shape[0]):
#             try:
#                 if different_shape_flag:
#                     sense_data_rotated[ch_idx,...] = rotate_image(sense_data[ch_idx,::-1,...],img_quat)
#                 else:
#                     sense_data_rotated[ch_idx,...] = rotate_image(sense_data[ch_idx,...,::-1],img_quat)
#             except:
#                 sense_data_rotated = np.zeros((sense_data.shape[0],sense_data.shape[-1],sense_data.shape[-2]),dtype=np.complex64)
#                 sense_data_rotated[ch_idx,...] = rotate_image(sense_data[ch_idx,::-1,...],img_quat)
#                 different_shape_flag = True

#         return sense_data_rotated

# def correction_map_rotation_old(auto_rotation, dim_info_org, data, num_sli, correction_map_all
#                             , inversed_correction_map_all, n, img_quat, correction_map, inversed_correction_map):
#     if auto_rotation == 'Dicom':
#         correction_map = np.rot90(correction_map,-1)
#         inversed_correction_map = np.rot90(inversed_correction_map,-1)

#     try:
#         if auto_rotation == 'Dicom':
#             correction_map_all[n,...] = rotate_image(correction_map,img_quat)
#             inversed_correction_map_all[n,...] = rotate_image(inversed_correction_map,img_quat)
#         else:
#             correction_map_all[n,...] = correction_map
#             inversed_correction_map_all[n,...] = inversed_correction_map
            
#     except:
#         if auto_rotation == 'Dicom':
#             correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
#             correction_map_all[n,...] = rotate_image(correction_map,img_quat)
#             inversed_correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
#             inversed_correction_map_all[n,...] = rotate_image(inversed_correction_map,img_quat)
#         else:
#             correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
#             correction_map_all[n,...] = correction_map
#             inversed_correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
#             inversed_correction_map_all[n,...] = inversed_correction_map
#     return correction_map_all,inversed_correction_map_all

def getting_and_saving_correction_map(base_dir ,input_folder, output_folder, folder_names = None, 
                           auto_rotation = 'Dicom',debug = True,apply_correction_during_sense_recon = False,
                           CustomProcedure = None, thread_idx = ''):
    input_folder = os.path.join(base_dir, input_folder)
    output_folder = os.path.join(base_dir, output_folder)
    folder_names = os.listdir(input_folder) if folder_names is None else folder_names
    for folder_name in folder_names:
    # Construct the full path to the directory
        full_dir_name = os.path.join(input_folder , folder_name)
        full_dir_name_output = os.path.join(output_folder , folder_name)
    
    # Check if it's a directory
        if os.path.isdir(full_dir_name):
        # Loop over all files in the directory
            for filename in os.listdir(full_dir_name):
            # Check if the file is a .dat file
                if filename.endswith(".dat"):
            #if filename.find('SAX') != -1:#this is a file filter uncomment this line and change the SAX to the keyword you want to filter
                # Construct the full file path
                    #full_file_path = os.path.join(full_dir_name, filename)
                    recon_results, correction_map_all, quat, inversed_correction_map_all, apply_correction_during_sense_recon = brightness_correction_map_generator(full_dir_name, filename, 
                                                                                                      auto_rotation,apply_correction_during_sense_recon,
                                                                                                      CustomProcedure)
                    if auto_rotation == 'LGE':
                        recon_results, correction_map_all, inversed_correction_map_all = rotate_images_for_LGE(recon_results, correction_map_all,inversed_correction_map_all, quat, filename)
                
                #print(filename)
                #print(correction_map_all.shape)
                # Save the results
                    if apply_correction_during_sense_recon:
                        corrected_results_filename = os.path.join(full_dir_name_output, f"{filename}.corrected_sense_results.npy")
                    else:
                        uncorrected_results_filename = os.path.join(full_dir_name_output, f"{filename}.uncorrected_results.npy")

                    correction_map_all_filename = os.path.join(full_dir_name_output, f"{filename}.image_correction_map.npy")
                    inversed_correction_map_all_filename = os.path.join(full_dir_name_output, f"{filename}.sensitivity_correction_map.npy")
                    quat_filename = os.path.join(full_dir_name_output, f"{filename}.quat.npy")
                # Save grappa_results and correction_map_all to their respective files
                    os.makedirs(full_dir_name_output, exist_ok=True)
                    if apply_correction_during_sense_recon:
                        np.save(corrected_results_filename, recon_results)
                    else:
                        np.save(uncorrected_results_filename, recon_results)
                    np.save(correction_map_all_filename, correction_map_all)
                    np.save(inversed_correction_map_all_filename, inversed_correction_map_all)
                    
                    #save quat and slc_dir for future debugging
                    if debug:
                        np.save(quat_filename, quat)
    #print end information with color
    print("\033[92m" + "Thread "+str(thread_idx)+" is done! You can find the results in the output folder! You can also display the results by using the function displaying_results(). Please check cells below!" + "\033[0m")


def divide_list(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]

import threading

def create_and_start_threadings(num_thread , target, base_dir, input_folder, output_folder, folder_names, 
                                auto_rotation= True,debug = True,apply_correction_during_sense_recon = False,
                                CustomProcedure = None):
   #create the output folder if it doesn't exist
   os.makedirs(os.path.join(base_dir, output_folder), exist_ok=True)

   #create_and_start_threadings
   folder_names = list(divide_list(folder_names, len(folder_names)//num_thread))

   threads = {}
   for i in range(len(folder_names)):
      threads[i] = threading.Thread( target = target, args=(base_dir, input_folder, output_folder, folder_names[i],
                                                            auto_rotation, debug, apply_correction_during_sense_recon,
                                                            CustomProcedure, i) )
      #print start information with green color
      print("\033[92m" + "Thread "+str(i)+" has been created! please wait until it finishes!" + "\033[0m")
      threads[i].start()

   return threads


def img_normalize(uncorrected_img):
    uncorrected_img = np.abs(uncorrected_img)
    uncorrected_img = uncorrected_img/np.max(uncorrected_img)
    corrected_img = uncorrected_img/np.percentile(uncorrected_img.flatten(),96)
    return corrected_img

import matplotlib.gridspec as gridspec

def displaying_results(base_dir ,input_folder, output_folder, folder_names = None, sli_idx = 0,par_idx = None, avg_idx = None, fig_h = 9, show_both = False):
    input_folder = os.path.join(base_dir, input_folder)
    output_folder = os.path.join(base_dir, output_folder)
    folder_names = os.listdir(input_folder) if folder_names is None else folder_names
    for file_name in folder_names:#['1','2','3','4','5','6','7','8','9','10','11','14','16','17','18','19']:#os.listdir(input_dir):
    # Construct the full path to the directory
        full_dir_name = os.path.join(input_folder, file_name)
        full_dir_name_output = os.path.join(output_folder, file_name)

        if os.path.isdir(full_dir_name):
        # Loop over all files in the directory
            for filename in os.listdir(full_dir_name):
                #if filename.find('SAX') != -1: #this is a file filter uncomment this line and change the SAX to the keyword you want to filter
                # Construct the full file path
                    #skip if the file starts with .ipynb_checkpoints.
                    if filename.startswith('.ipynb_checkpoints'):
                        continue
                
                    filename = filename[:-4]

                    try:
                        correction_map_all_filename = os.path.join(full_dir_name_output, f"{filename}.image_correction_map.npy")
                        correction_map_all = np.load(correction_map_all_filename, allow_pickle=True)
                        if len(correction_map_all.shape) == 4:
                            if par_idx is not None:
                                correction_map_all = correction_map_all[...,par_idx]
                            else:
                                #throw an error if par_idx is None and len(correction_map_all.shape) == 4
                                raise ValueError("par_idx should not be None if Par dimension exists")
                    except:
                        ...
                    try:
                        inversed_correction_map_all_filename = os.path.join(full_dir_name_output, f"{filename}.sensitivity_correction_map.npy")
                        inversed_correction_map_all = np.load(inversed_correction_map_all_filename)
                        if len(inversed_correction_map_all.shape) == 4:
                            if par_idx is not None:
                                inversed_correction_map_all = inversed_correction_map_all[...,par_idx]
                            else:
                                #throw an error if par_idx is None and len(inversed_correction_map_all.shape) == 4
                                raise ValueError("par_idx should not be None if Par dimension exists")
                    except:
                        ...

                    try:
                        corrected_results_filename = os.path.join(full_dir_name_output, f"{filename}.corrected_sense_results.npy")
                        corrected_results = np.load(corrected_results_filename, allow_pickle=True)
                        if len(corrected_results.shape) == 4:
                            if par_idx is not None:
                                corrected_results = corrected_results[...,par_idx]
                            else:
                                #throw an error if par_idx is None and len(corrected_results.shape) == 4
                                raise ValueError("par_idx should not be None if Par dimension exists")
                    except:
                        corrected_results = None
                    try:
                        uncorrected_results_filename = os.path.join(full_dir_name_output, f"{filename}.uncorrected_results.npy")
                        uncorrected_results = np.load(uncorrected_results_filename, allow_pickle=True)
                        if len(uncorrected_results.shape) == 4:
                            if par_idx is not None:
                                uncorrected_results = uncorrected_results[...,par_idx]
                            else:
                                #throw an error if par_idx is None and len(uncorrected_results.shape) == 4
                                raise ValueError("par_idx should not be None if Par dimension exists")
                    except:
                        uncorrected_results = None

                    try:
                        quat_filename = os.path.join(full_dir_name_output, f"{filename}.quat.npy")
                        quat = np.load(quat_filename, allow_pickle=True)
                        *_, slc_dir_vec = quaternion_to_directions(quat)

                    except:
                        ...

                    # #import datetime    
                    # import datetime
                    # #get current time as time stamp
                    # time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    # #save the images
                    # #first image np.abs(correction_map_all[sli_idx,...])
                    # plt.figure()
                    # plt.imshow(np.abs(correction_map_all[sli_idx,...]),cmap='jet')
                    # plt.axis("off")
                    # plt.savefig(os.path.join("./", f"{filename}.image_correction_map_{time_stamp}.png"), bbox_inches='tight', pad_inches=0, dpi=3000)
                    # #second image np.abs(inversed_correction_map_all[sli_idx,...])
                    # plt.figure()
                    # plt.imshow(np.abs(inversed_correction_map_all[sli_idx,...]),cmap='jet')
                    # plt.axis("off")
                    # plt.savefig(os.path.join("./", f"{filename}.sensitivity_correction_map_{time_stamp}.png"), bbox_inches='tight', pad_inches=0, dpi=3000)
                    # #third image np.abs(uncorrected_results[sli_idx,...])
                    # if uncorrected_results is not None:
                    #     plt.figure()
                    #     plt.imshow(np.abs(img_normalize(uncorrected_results[sli_idx,...])),cmap='gray',vmax=1)
                    #     plt.axis("off")
                    #     plt.savefig(os.path.join("./", f"{filename}.uncorrected_results_{time_stamp}.png"), bbox_inches='tight', pad_inches=0, dpi=3000)
                    # #fourth image np.multiply(correction_map_all[sli_idx,...],uncorrected_results[sli_idx,...])
                    # if uncorrected_results is not None:
                    #     plt.figure()
                    #     plt.imshow(img_normalize(np.multiply(correction_map_all[sli_idx,...],uncorrected_results[sli_idx,...])),cmap='gray',vmax=1)
                    #     plt.axis("off")
                    #     plt.savefig(os.path.join("./", f"{filename}.corrected_results_{time_stamp}.png"), bbox_inches='tight', pad_inches=0, dpi=3000)
                    # #fifth image img_normalize(corrected_results[sli_idx,...])
                    # if corrected_results is not None:
                    #     plt.figure()
                    #     plt.imshow(img_normalize(corrected_results[sli_idx,...]),cmap='gray',vmax=1)
                    #     plt.axis("off")
                    #     plt.savefig(os.path.join("./", f"{filename}.corrected_results_(sensitivity){time_stamp}.png"), bbox_inches='tight', pad_inches=0, dpi=3000)
                    # continue


                    img_shape = correction_map_all.shape[-2:]
                    if img_shape[0] > img_shape[1]:
                        auto_width = int(( np.max(img_shape) / np.min(img_shape) )*fig_h)
                    else:
                        auto_width = 18


                    if show_both:

                        if (correction_map_all.shape[0]-1)>=sli_idx:
                            #print(results_filename)
                            if uncorrected_results is not None:
                                ncols = 6
                                width_ratios = [0.5, 9, 9, 9, 9, 1]

                                if corrected_results is not None:
                                    ncols = 7
                                    width_ratios = [0.5, 9, 9, 9, 9, 9, 1]

                            elif corrected_results is not None:
                                ncols = 5
                                width_ratios = [0.5, 9, 9, 9, 1]

                            try:
                                #subdivide number 
                                subdevide_num = 40

                                # Create a GridSpec object
                                gs = gridspec.GridSpec(subdevide_num, ncols=ncols, width_ratios=width_ratios, figure=plt.figure(figsize=(auto_width,fig_h))) # Adjust the figure size as needed

                                # Create subplots and colorbar axes
                                
                                ax1 = plt.subplot(gs[0:subdevide_num, 1])
                                ax2 = plt.subplot(gs[0:subdevide_num, 2])

                                im1 = ax1.imshow(np.abs(correction_map_all[sli_idx,...]),cmap='jet')
                                ax1.axis("off")
                                ax1.set_title("image correction map")

                                im2 = ax2.imshow(np.abs(inversed_correction_map_all[sli_idx,...]),cmap='jet')
                                ax2.axis("off")
                                ax2.set_title("sensetivity correction map")

                                if uncorrected_results is not None:
                                    delta_for_colorbar = 9

                                    ax3 = plt.subplot(gs[0:subdevide_num, 3])
                                    ax4 = plt.subplot(gs[0:subdevide_num, 4])

                                    uncorrected_img = img_normalize(uncorrected_results[sli_idx,...])
                                    im3 = ax3.imshow(uncorrected_img**1,vmax=1,cmap='gray')
                                    ax3.axis("off")
                                    ax3.set_title("uncorrected image")

                                    corrected_img = np.multiply(correction_map_all[sli_idx,...],uncorrected_results[sli_idx,...])
                                    corrected_img = img_normalize(corrected_img)
                                    im4 = ax4.imshow(corrected_img**1,cmap='gray',vmax=1)
                                    ax4.axis("off")
                                    ax4.set_title("corrected with\nimage correction map")

                                    if corrected_results is not None:
                                        ax5 = plt.subplot(gs[0:subdevide_num, 5])
                                        corrected_img = img_normalize(corrected_results[sli_idx,...])
                                        im5 = ax5.imshow(corrected_img**1,cmap='gray',vmax=1)
                                        ax5.axis("off")
                                        ax5.set_title("corrected with\nsensitivity correction map")

                                    try:
                                        plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + '\n' +"quaternions: "+str(quat)+"\n"+"slice vector: "+str(slc_dir_vec)+"\n\n", x=0.5, y=0.80, ha='center')
                                    except:
                                        plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + "\n\n", x=0.5, y=0.80, ha='center')

                                elif corrected_results is not None:
                                    delta_for_colorbar = 1

                                    ax3 = plt.subplot(gs[0:subdevide_num, 3])
                                    corrected_img = img_normalize(corrected_results[sli_idx,...])
                                    im3 = ax3.imshow(corrected_img**1,cmap='gray',vmax=1)
                                    ax3.axis("off")
                                    ax3.set_title("corrected with\nsensitivity correction map")

                                    try:
                                        plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + '\n' +"quaternions: "+str(quat)+"\n"+"slice vector: "+str(slc_dir_vec)+"\n\n", x=0.5, y=1.0, ha='center')
                                    except:
                                        plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + "\n\n", x=0.5, y=1.0, ha='center')

                                cax1 = plt.subplot(gs[delta_for_colorbar:subdevide_num-delta_for_colorbar, 0])
                                # Display colorbar for the first subplot
                                plt.colorbar(im1, cax=cax1)
                                plt.tight_layout();plt.show()

                            except Exception as e:
                                print(e)
                                ...
                    else:
                        if (correction_map_all.shape[0]-1)>=sli_idx:
                            #print(grappa_results_filename)
                            try:
                                # Create a GridSpec object
                                gs = gridspec.GridSpec(50, 5, width_ratios=[0.5, 9, 9, 9, 1], figure=plt.figure(figsize=(auto_width,fig_h))) # Adjust the figure size as needed

                                # Create subplots and colorbar axes
                                ax1 = plt.subplot(gs[0:50, 1])
                                cax1 = plt.subplot(gs[2:48, 0])
                                ax2 = plt.subplot(gs[0:50, 2])
                                ax3 = plt.subplot(gs[0:50, 3])

                                im1 = ax1.imshow(np.abs(correction_map_all[sli_idx,...]),cmap='jet')
                                ax1.axis("off")
                                ax1.set_title("image correction map")
                                

                                uncorrected_img = img_normalize(uncorrected_results[sli_idx,...])
                                im2 = ax2.imshow(uncorrected_img**1,vmax=1,cmap='gray')
                                ax2.axis("off")
                                ax2.set_title("before correction")
                                
                                
                                corrected_img = np.multiply(correction_map_all[sli_idx,...],uncorrected_results[sli_idx,...])
                                corrected_img = img_normalize(corrected_img)
                                im3 = ax3.imshow(corrected_img**1,cmap='gray',vmax=1)
                                ax3.axis("off")
                                ax3.set_title("after correction")
                                
                                # Display colorbar for the first subplot
                                plt.colorbar(im1, cax=cax1)
                                try:
                                    plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + '\n' +"quaternions: "+str(quat)+"\n"+"slice vector: "+str(slc_dir_vec))
                                except:
                                    plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename)

                                plt.tight_layout();plt.show()
                            except Exception as e:
                                print(e)
                                ...