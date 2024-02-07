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



# Constants for image types
# In order to determine the view type, the file name must contain the keyword of the view type
LAX = 'LAX'
SAX = 'SAX'
CH2 = '2CH'

def rotate_images_for_LGE(results, quat, filename):
    """
    Rotates and flips images for LGE based on the orientation and type.
    
    :param results: numpy array of the image to be processed
    :param quat: quaternion representing the image orientation
    :param filename: name of the file, used to determine the type of image
    :return: Processed numpy array of the image
    """
    if results is None:
        return None

    read_dir, phase_dir, _ = quaternion_to_directions(quat)

    # Rotate and flip images for LAX view
    if LAX in filename:
        if read_dir[0] > 0:
            results = np.flip(results, axis=2)          #flip the image
        results = np.rot90(results, k=-1, axes=(1, 2))  #rotate the image clockwise 90 degree

    # Rotate and flip images for SAX view
    elif SAX in filename:
        if read_dir[-1] > 0:
            results = np.flip(results, axis=2)          #flip the image
        if phase_dir[0] < 0:
            results = np.flip(results, axis=1)          #flip the image
        results = np.rot90(results, axes=(1, 2))        #rotate the image

    # Rotate and flip images for 2CH view
    elif CH2 in filename:
        if read_dir[-1] > 0:
            results = np.flip(results, axis=2)
        if phase_dir[0] < 0 and phase_dir[1] > 0:
            results = np.flip(results, axis=1)
        results = np.rot90(results, axes=(1, 2))

    return results

def get_dimension_indices(all_dimensions, dimensions_to_find):
    """
    Finds the indices of specified dimensions in a list of dimensions.

    :param all_dimensions: List of all dimensions.
    :param dimensions_to_find: List of dimensions to find.
    :return: List of indices of the found dimensions.
    """
    indices = []
    for dim in dimensions_to_find:
        try:
            index = all_dimensions.index(dim)
            indices.append(index)
        except ValueError:
            print(f"Dimension '{dim}' not found, skipping.")
    return indices

def move_dimension_to_front_data(array, all_dimensions, dim_to_move):
    """
    Moves a specified dimension to the front of a multi-dimensional array.

    :param array: The array to be manipulated.
    :param all_dimensions: List of all dimensions in the array.
    :param dim_to_move: The dimension to move to the front.
    :return: Array with the specified dimension moved to the front.
    """
    try:
        source = get_dimension_indices(all_dimensions, [dim_to_move])[0]
        return np.moveaxis(array, source, 0)
    except IndexError:
        print(f"Warning: Dimension '{dim_to_move}' not found.")
        return array
    
def move_dimension_to_front_info(all_dimensions, dim_to_move):
    """
    Moves a specified dimension to the front of a dimensions list.

    :param all_dimensions: List of dimensions.
    :param dim_to_move: The dimension to move to the front.
    :return: Updated list of dimensions with the specified dimension moved to the front.
    """
    if dim_to_move in all_dimensions:
        all_dimensions.remove(dim_to_move)
        all_dimensions.insert(0, dim_to_move)
    else:
        print(f"Warning: Dimension '{dim_to_move}' not found.")
    return all_dimensions
    

def data_reduction(data, data_dimensions, dims_to_keep=['Sli', 'Lin', 'Cha', 'Col'], dim_to_set_to_zero=['Phs', 'Set']):
    """
    Reduces data based on specified dimensions to keep and dimensions to set to zero.
    All possible dimensions: ['Ide', 'Idd', 'Idc', 'Idb', 'Ida', 'Seg', 'Set', 'Rep','Phs', 'Eco', 'Par', 'Sli', 'Ave', 'Lin', 'Cha', 'Col']

    :param data: Multi-dimensional data array.
    :param data_dimensions: List of dimensions in the data array.
    :param dims_to_keep: Dimensions to keep in the reduced data.
    :param dim_to_set_to_zero: Dimensions to set to zero.
    :return: Tuple containing the reduced data array and updated dimensions list.
    """
    for dim in dim_to_set_to_zero:
        if dim in data_dimensions:
            data = move_dimension_to_front_data(data, data_dimensions, dim)
            data_dimensions = move_dimension_to_front_info(data_dimensions, dim)
            data = data[0, ...]   # Set the first slice to zero
            data_dimensions = data_dimensions[1:]   #remove the first element in data_dimensions

    shape = data.shape
    slices = [slice(None)] * len(shape)
    dims_found = get_dimension_indices(data_dimensions, dims_to_keep)

    for i in range(len(shape)):
        if i not in dims_found:
            # Only slice the dimensions that are not specified in dims_to_keep
            middle_index = shape[i] // 2   # This is the middle index
            slices[i] = slice(middle_index, middle_index + 1)   # Get the middle slice

    data_dimensions = [dim for i, dim in enumerate(data_dimensions) if i in dims_found]

    return np.squeeze(data[tuple(slices)]), data_dimensions   # Squeeze to remove the dimensions that have a length of 1




def target_path_generator(base_dir, input_folder, output_folder, input_subfolders=None):
    """
    Generates paths for input and output data based on the specified base directory, 
    input and output folders, and subfolders.

    :param base_dir: Base directory for input and output folders.
    :param input_folder: Name of the input folder.
    :param output_folder: Name of the output folder.
    :param input_subfolders: List of subfolders within the input folder. 
                             If None, all subfolders are included.
    :return: Tuple of lists containing the paths to the input and output data files.
    """
    data_path_names = []
    data_path_names_output = []
    input_folder = os.path.join(base_dir, input_folder)
    output_folder = os.path.join(base_dir, output_folder)
    input_subfolders = os.listdir(input_folder) if input_subfolders is None else input_subfolders

    for input_subfolder in input_subfolders:
        # Construct the full path to the directory
        full_dir_name = os.path.join(input_folder, input_subfolder)
        full_dir_name_output = os.path.join(output_folder, input_subfolder)

        # Check if it's a directory
        if os.path.isdir(full_dir_name):
            # Loop over all files in the directory
            for filename in os.listdir(full_dir_name):
                # Check if the file is a .dat file
                if filename.endswith((".dat", ".gz")):
                    data_path_names.append(os.path.join(full_dir_name, filename))
                    data_path_names_output.append(os.path.join(full_dir_name_output, filename))

    return data_path_names, data_path_names_output



def rawdata_reader(data_path_filename):
    """
    Reads raw data from a given file path.
    if data_path_filename ends with .demo.gz, then it is a demo file otherwise it is a rawdata file
    the code can handle both cases

    :param data_path_filename: Path to the data file.
    :return: Multiple data items including 'twix', 'image_3D_body_coils', 
             'image_3D_surface_coils', 'data', 'dim_info_data', 'data_ref', 
             'dim_info_ref', and 'num_sli'.
    """
    try:
        twix, mapped_data, data_org, dim_info_org, data_ref, dim_info_ref, noise_kspace, dim_info_noise = readtwix_arry_all(data_path_filename = data_path_filename)

        data = data_org.squeeze()   # this will squeeze the dimensions
        dim_info_data = dim_info_org.copy()
        # end of reading the data

        # unpack the mapped_data
        image_3D_body_coils, image_3D_surface_coils = generate_3D_data(mapped_data)

        num_sli = data.shape[dim_info_org.index('Sli')] if 'Sli' in dim_info_org else 1

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

    print('Number of slices in the rawdata: ', num_sli)

    return twix, image_3D_body_coils, image_3D_surface_coils, data, dim_info_data, data_ref, dim_info_ref, num_sli


def low_resolution_img_interpolator(twix, image_3D_body_coils, image_3D_surface_coils, data, dim_info_data, num_sli, auto_rotation='Dicom'):
    """
    Interpolates low-resolution MRI images.

    Parameters:
    - twix: Raw data from the MRI scanner.
    - image_3D_body_coils: 3D image data from body coils.
    - image_3D_surface_coils: 3D image data from surface coils.
    - data: Extracted image data.
    - dim_info_data: Information about the dimensions of the data.
    - num_sli: Number of slices in the data.
    - auto_rotation: Rotation mode for image correction (default 'Dicom').

    Returns:
    - img_correction_map_all: Image correction maps for all slices.
    - sensitivity_correction_map_all: Sensitivity correction maps for all slices.
    - low_resolution_surface_coil_imgs: Interpolated low-resolution images.
    - img_quats: Quaternion data for image orientation.
    """
    lin_index = dim_info_data.index('Lin')
    col_index = dim_info_data.index('Col')

    img_correction_map_all = np.zeros((num_sli, data.shape[lin_index], data.shape[col_index] // 2))
    sensitivity_correction_map_all = np.zeros_like(img_correction_map_all)
    low_resolution_surface_coil_imgs = np.zeros_like(img_correction_map_all, dtype=np.complex64)
    img_quats = []

    for n in range(num_sli):
        result = calculating_correction_maps(auto_rotation, twix, dim_info_data, data, 
                                             image_3D_body_coils, image_3D_surface_coils, 
                                             num_sli, img_correction_map_all, 
                                             sensitivity_correction_map_all, n)

        img_correction_map_all, img_quat, x2d, sensitivity_correction_map_all = result
        low_resolution_surface_coil_imgs[n, ...] = x2d
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
    oversampling_phase_factor: int = 1,
    remove_readout_oversampling: bool = False):

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
    - remove_readout_over_sampling: Flag to remove readout oversampling. When True the shape of low resolution images: 64x64x64, when False,the shape is 128x64x64  (default is False).

    Returns:
    - img_correction_map_all: Array of image correction maps for each slice.
    - sensitivity_correction_map_all: Array of sensitivity correction maps for each slice.
    - low_resolution_surface_coil_imgs: Low-resolution images from surface coils.
    - img_quats: A list containing quaternions for each slice.

    Exceptions:
    - Catches and handles exceptions related to data shape and dimension indexing, defaulting to specific array 
      configurations in case of an error.
    """

    lin_index = dim_info_data.index('Lin')
    col_index = dim_info_data.index('Col')

    try:
        par_num = data.shape[dim_info_data.index('Par')]
        shape = (num_sli, data.shape[lin_index], data.shape[col_index] // 2, par_num) if par_num > 1 else (num_sli, data.shape[lin_index], data.shape[col_index] // 2)
    except:
        shape = (num_sli, data.shape[lin_index], data.shape[col_index] // 2)

    img_correction_map_all = np.zeros(shape)
    sensitivity_correction_map_all = np.zeros_like(img_correction_map_all)
    low_resolution_surface_coil_imgs = np.zeros_like(img_correction_map_all, dtype=np.complex64)

    img_quats = []
    correction_map3D, correction_map3D_sense = None, None

    for n in range(num_sli):
        result = calculating_correction_maps(auto_rotation, twix, dim_info_data, data, 
                                             image_3D_body_coils, image_3D_surface_coils, 
                                             num_sli, img_correction_map_all, 
                                             sensitivity_correction_map_all, 
                                             low_resolution_surface_coil_imgs, n, 
                                             lamb=lamb, tol=tol, maxiter=maxiter,
                                             apply_correction_to_sensitivity_maps=apply_correction_to_sensitivity_maps,
                                             correction_map3D=correction_map3D, 
                                             correction_map3D_sense=correction_map3D_sense, 
                                             oversampling_phase_factor=oversampling_phase_factor,
                                             remove_readout_oversampling=remove_readout_oversampling)

        img_correction_map_all, img_quat, low_resolution_surface_coil_imgs, sensitivity_correction_map_all, correction_map3D, correction_map3D_sense = result
        img_quats.append(img_quat)

    return img_correction_map_all, sensitivity_correction_map_all, low_resolution_surface_coil_imgs, img_quats


def auto_image_rotation(sense_recon_results, img_quats, auto_rotation='Dicom', img_correction_map=None, sens_correction_map=None, filename=None):
    """
    Automatically rotates image reconstruction results based on the specified rotation mode.

    Parameters:
    - sense_recon_results: Reconstructed sense images.
    - img_quats: Image quaternions for orientation.
    - auto_rotation: Rotation mode, either 'Dicom' or 'LGE'.
    - img_correction_map: Image correction map.
    - sens_correction_map: Sensitivity correction map.
    - filename: Filename for determining specific rotation in 'LGE' mode.

    Returns:
    - Tuple of rotated sense reconstruction results, image correction map, and sensitivity correction map.
    """
    valid_rotations = ['Dicom', 'LGE']
    if auto_rotation not in valid_rotations:
        raise ValueError(f"auto_rotation should be either 'Dicom' or 'LGE'")

    if auto_rotation == 'Dicom':
        #num_sli = sense_recon_results.shape[0]
        #for n in range(num_sli):
        #    sense_recon_results_rotated = sense_img_rotation(sense_recon_results_rotated, sense_recon_results[n,...], img_quats[n], num_sli, n, auto_rotation = auto_rotation)
        #return sense_recon_results_rotated, img_correction_map, sens_correction_map
        return sense_recon_results, img_correction_map, sens_correction_map

    if auto_rotation == 'LGE':
        sense_recon_results_rotated = rotate_images_for_LGE(sense_recon_results, img_quats[0], filename)
        img_correction_map_rotated = rotate_images_for_LGE(img_correction_map, img_quats[0], filename)
        sens_correction_map_rotated = rotate_images_for_LGE(sens_correction_map, img_quats[0], filename)
        return sense_recon_results_rotated, img_correction_map_rotated, sens_correction_map_rotated

                

def sense_img_rotation(sense_recon_results, sense_reconstructed_img, img_quat, num_sli, n, auto_rotation='Dicom'):
    """
    Rotates a sense reconstructed image based on the specified rotation mode.

    Parameters:
    - sense_recon_results: Array to store rotated images.
    - sense_reconstructed_img: Image to be rotated.
    - img_quat: Quaternion for image rotation.
    - num_sli: Total number of slices.
    - n: Current slice number.
    - auto_rotation: Rotation mode.

    Returns:
    - Array with the rotated image for the current slice.
    """
    if auto_rotation == 'Dicom':
        sense_reconstructed_img = np.rot90(sense_reconstructed_img, -1)
        sense_reconstructed_img = rotate_image(sense_reconstructed_img, img_quat)

    try:
        sense_recon_results[n, ...] = sense_reconstructed_img
    except ValueError:
        # Resize sense_recon_results if the shape does not match
        new_shape = (num_sli, sense_reconstructed_img.shape[-2], sense_reconstructed_img.shape[-1])
        sense_recon_results = np.zeros(new_shape, dtype=np.complex64)
        sense_recon_results[n, ...] = sense_reconstructed_img

    return sense_recon_results
            

def save_sense_recon_results(full_dir_name_output, sense_recon_results, img_correction_map, sens_correction_map, quat, apply_correction_during_sense_recon):
    """
    Saves sense reconstruction results and correction maps to files.

    Parameters:
    - full_dir_name_output: Base directory for output files.
    - sense_recon_results: Reconstructed sense images.
    - img_correction_map: Image correction map.
    - sens_correction_map: Sensitivity correction map.
    - quat: Quaternion data.
    - apply_correction_during_sense_recon: Flag to indicate if correction was applied during reconstruction.
    """
    base_filename = full_dir_name_output[:-4]
    os.makedirs(os.path.dirname(base_filename), exist_ok=True)

    if apply_correction_during_sense_recon:
        np.save(base_filename + ".corrected_sense_results.npy", sense_recon_results)
    else:
        np.save(base_filename + ".uncorrected_results.npy", sense_recon_results)

    if np.sum(img_correction_map) != 0 and img_correction_map is not None and np.any(~np.isnan(img_correction_map)):
        np.save(base_filename + ".image_correction_map.npy", img_correction_map)
    if np.sum(sens_correction_map) != 0 and sens_correction_map is not None and np.any(~np.isnan(sens_correction_map)):
        np.save(base_filename + ".sensitivity_correction_map.npy", sens_correction_map)

    try:
        np.save(base_filename + ".quat.npy", quat[0])   #save quat and slc_dir for future debugging
    except Exception:
        pass  # Handle potential errors in saving quat




def single_img_interpolation(twix, image_3D_body_coils, image_3D_surface_coils, num_sli, n):
    """
    Performs interpolation for a single image in MRI data.

    Parameters:
    - twix: Raw data from the MRI scanner.
    - image_3D_body_coils: 3D image data from body coils.
    - image_3D_surface_coils: 3D image data from surface coils.
    - num_sli: Number of slices.
    - n: Slice index.

    Returns:
    - Interpolated images from body and surface coils, and image quaternion.
    """
    Zi_body_coils, Zi_surface_coils, img_quat, _ = interpolation(twix, image_3D_body_coils, image_3D_surface_coils, num_sli, n)
    inter_img_body_coils, inter_img_surface_coils = remove_edges(Zi_body_coils, Zi_surface_coils)
    inter_img_body_coils = inter_img_body_coils.transpose([2, 0, 1])
    inter_img_surface_coils = inter_img_surface_coils.transpose([2, 0, 1])

    return inter_img_body_coils, inter_img_surface_coils, img_quat





def single_correction_map_generator(auto_rotation, img_quat, dim_info_org, data, inter_img_body_coils, 
                                    inter_img_surface_coils, num_sli, correction_map_all, 
                                    inversed_correction_map_all, n):
    """
    Generates a single correction map for MRI images.

    Parameters:
    - auto_rotation: Rotation mode.
    - img_quat: Image quaternion for orientation.
    - dim_info_org: Original dimension information.
    - data: MRI data.
    - inter_img_body_coils: Interpolated images from body coils.
    - inter_img_surface_coils: Interpolated images from surface coils.
    - num_sli: Number of slices.
    - correction_map_all: Existing correction maps.
    - inversed_correction_map_all: Existing inversed correction maps.
    - n: Slice index.

    Returns:
    - Updated correction maps, image quaternion, and interpolated 2D image.
    """
    inter_img_body_coils = rms_comb(inter_img_body_coils, 0)
    inter_img_surface_coils = rms_comb(inter_img_surface_coils, 0)

    x2d = normalize_image(inter_img_surface_coils)  # Normalized 2D image
    x3d = normalize_image(inter_img_body_coils)  # Normalized 3D image

    # Calculating correction maps
    _, _, correction_map = calculate_correction_map(A=x2d, B=x3d, lamb=1e-1)
    _, _, inversed_correction_map = calculate_correction_map(A=x3d, B=x2d, lamb=1e-1)

    correction_map_all, inversed_correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                                              correction_map_all, inversed_correction_map_all, 
                                                                              n, img_quat, correction_map, inversed_correction_map)
    return correction_map_all, img_quat, x2d, inversed_correction_map_all




import cv2


# Normalize and convert the images to uint8
def normalize_convert_uint8(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    uint8_image = (normalized_image * 255).astype('uint8')
    return uint8_image


def find_and_combine_contours(image_3D_body_coils, image_3D_surface_coils):
    """
    Finds and combines contours from 3D body and surface coil images.

    Parameters:
    - image_3D_body_coils: 3D image from body coils.
    - image_3D_surface_coils: 3D image from surface coils.

    Returns:
    - Combined contour image and the contours.
    """
    # Process the image and find edges
    def find_edges(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary_image, 50, 150)
        return cv2.dilate(edges, None, iterations=1)

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
    combined_edges = (edges_body_coils // 2 + edges_surface_coils // 2).astype(np.uint8)

    # Find the combined contours and draw the combined contours on a blank image
    combined_contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    combined_contour_image = draw_contours_on_blank(image_3D_body_coils_uint8, combined_contours)

    return combined_contour_image, combined_contours



def create_energy_based_mask(image, contours):
    """
    Creates an energy-based mask for an MRI image using contours.

    Parameters:
    - image: MRI image.
    - contours: Contours found in the image.

    Returns:
    - Binary mask based on energy levels inside and outside the contours.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        if contour.size > 0:
            contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)
            
            inside_average = np.mean(image[contour_mask == 1])
            outside_average = np.mean(image[contour_mask == 0])
            
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

    combined_contour_image, combined_contours = find_and_combine_contours(image_3D_body_coils_3D, image_3D_surface_coils_3D)

    mask = create_energy_based_mask(image_3D_body_coils_3D+image_3D_surface_coils_3D, combined_contours)
    combined_contour_image = combined_contour_image.astype(np.float64)

    #plot image_3D_body_coils_3D and image_3D_surface_coils_3D
    # img_temp = image_3D_body_coils_3D+(combined_contour_image/255)*np.max(image_3D_body_coils_3D)
    # img_temp = image_3D_surface_coils_3D+(combined_contour_image/255)*np.max(image_3D_surface_coils_3D)


    # Apply combined mask to correction_map3D
    corrected_map = correction_map3D * mask
    
    return corrected_map


def calculating_correction_maps(auto_rotation, twix, dim_info_org, 
                                data, image_3D_body_coils, image_3D_surface_coils, 
                                num_sli, correction_map_all,inversed_correction_map_all, low_resolution_surface_coil_imgs,n, lamb = 1e-3,tol = 1e-4, maxiter=500, 
                                apply_correction_to_sensitivity_maps = False, correction_map3D = None , correction_map3D_sense = None,oversampling_phase_factor = 1,
                                remove_readout_oversampling = False):


    # Initialize correction maps if not provided
    if correction_map3D is None and correction_map3D_sense is None:
        image_3D_body_coils_3D = rms_comb(image_3D_body_coils.copy(),-1)
        image_3D_surface_coils_3D = rms_comb(image_3D_surface_coils.copy(),-1)

        # print("image_3D_body_coils_3D",image_3D_body_coils_3D.shape)
        # print("image_3D_surface_coils_3D",image_3D_surface_coils_3D.shape)

        temp_maps = calculate_correction_map_3D(image_3D_surface_coils_3D,image_3D_body_coils_3D,lamb=lamb,tol=tol, 
                                                    maxiter=maxiter, sensitivity_correction_maps=apply_correction_to_sensitivity_maps, 
                                                    debug = False, remove_readout_oversampling = remove_readout_oversampling)

        if apply_correction_to_sensitivity_maps == False:
            correction_map3D       = temp_maps
        else:
            correction_map3D_sense = temp_maps
        

    # Interpolate data and process each slice or parameter set
    interpolated_data, img_quat, _ = interpolation(
        twix, data, dim_info_org, num_sli, n,
        [image_3D_body_coils, image_3D_surface_coils, correction_map3D, correction_map3D_sense],
        oversampling_phase_factor=oversampling_phase_factor
    )
    
    inter_img_body_coils, inter_img_surface_coils, correction_map_from_3D, correction_map_from_3D_sense = interpolated_data
    inter_img_surface_coils = rms_comb(inter_img_surface_coils.copy(), -1)
    inter_img_body_coils = rms_comb(inter_img_body_coils.copy(), -1)

    try:
        par_num = data.shape[dim_info_org.index('Par')]
    #if Par is not in the list
    except ValueError:
        par_num = 0
    except Exception as e:
        # Specific exception handling or logging should be implemented here
        print(f"An error occurred during processing: {e}")

    try:
        if par_num > 1:
            # Process for each parallel imaging parameter
            for par_idx in range(par_num):
                correction_args = (auto_rotation, dim_info_org, data, num_sli, n, apply_correction_to_sensitivity_maps,
                                   oversampling_phase_factor, img_quat, inter_img_body_coils[..., par_idx],inter_img_surface_coils[..., par_idx],

                                   correction_map_from_3D[..., par_idx] if not apply_correction_to_sensitivity_maps else None,
                                   correction_map_from_3D_sense[..., par_idx] if apply_correction_to_sensitivity_maps else None,
                                   correction_map_all[..., par_idx] if not apply_correction_to_sensitivity_maps else None, 
                                   inversed_correction_map_all[..., par_idx] if apply_correction_to_sensitivity_maps else None,

                                   low_resolution_surface_coil_imgs[..., par_idx])
                
                correction_map_all[...,par_idx], inversed_correction_map_all[...,par_idx], low_resolution_surface_coil_imgs[...,par_idx] = post_processing(*correction_args)
        else:
            # Single parameter set processing
            correction_map_all, inversed_correction_map_all, low_resolution_surface_coil_imgs = post_processing(
                                    auto_rotation, dim_info_org, data, num_sli, n, apply_correction_to_sensitivity_maps,
                                    oversampling_phase_factor, img_quat, inter_img_body_coils, inter_img_surface_coils,

                                    correction_map_from_3D, correction_map_from_3D_sense,
                                    correction_map_all, inversed_correction_map_all, 

                                    low_resolution_surface_coil_imgs)
    except Exception as e:
        # Specific exception handling or logging should be implemented here
        print(f"An error occurred during processing: {e}")
    
                                                                 
    return correction_map_all,img_quat,low_resolution_surface_coil_imgs,inversed_correction_map_all, correction_map3D, correction_map3D_sense



def post_processing(auto_rotation, dim_info_org, data, num_sli, n, 
                    apply_correction_to_sensitivity_maps, oversampling_phase_factor, 
                    img_quat, inter_img_body_coils, inter_img_surface_coils, 
                    correction_map_from_3D, correction_map_from_3D_sense,
                    correction_map_all, inversed_correction_map_all,
                    low_resolution_surface_coil_imgs):
    """
    Post-processes MRI data for correction map application and normalization.

    Parameters:
    - auto_rotation: Rotation mode, either 'Dicom' or 'LGE'.
    - dim_info_org: Original dimension information of the data.
    - data: MRI image data.
    - num_sli: Number of slices.
    - n: Current slice index.
    - apply_correction_to_sensitivity_maps: Flag for applying correction to sensitivity maps.
    - oversampling_phase_factor: Factor for phase oversampling.
    - img_quat: Image quaternion for orientation.
    - inter_img_body_coils: Interpolated images from body coils.
    - inter_img_surface_coils: Interpolated images from surface coils.
    - correction_map_from_3D: 3D correction map.
    - correction_map_from_3D_sense: 3D sense correction map.
    - correction_map_all: All correction maps.
    - inversed_correction_map_all: All inverse correction maps.
    - low_resolution_surface_coil_imgs: Low-resolution images from surface coils.

    Returns:
    - Updated correction maps and low-resolution surface coil images.
    """
    
    # Apply correction mask based on the oversampling phase factor and sensitivity map settings
    if oversampling_phase_factor != 1:
        if not apply_correction_to_sensitivity_maps:
            correction_map_from_3D = apply_correction_mask(inter_img_surface_coils, inter_img_body_coils, correction_map_from_3D)
        else:
            correction_map_from_3D_sense = apply_correction_mask(inter_img_surface_coils, inter_img_body_coils, correction_map_from_3D_sense)

    # Remove edges and oversampling phase direction
    correction_map_from_3D = remove_edges(correction_map_from_3D)
    correction_map_from_3D_sense = remove_edges(correction_map_from_3D_sense)
    correction_map_from_3D = remove_oversampling_phase_direction(correction_map_from_3D, oversampling_phase_factor)
    correction_map_from_3D_sense = remove_oversampling_phase_direction(correction_map_from_3D_sense, oversampling_phase_factor)

    # Process surface coil images
    inter_img_surface_coils = remove_edges(inter_img_surface_coils)
    inter_img_surface_coils = remove_oversampling_phase_direction(inter_img_surface_coils, oversampling_phase_factor)
    x2d = normalize_image(inter_img_surface_coils)

    # Determine which correction maps to use based on sensitivity map settings
    correction_map = correction_map_from_3D if not apply_correction_to_sensitivity_maps else None
    inversed_correction_map = correction_map_from_3D_sense if apply_correction_to_sensitivity_maps else None
    # Rotate and apply correction maps
    if not apply_correction_to_sensitivity_maps:
        correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                    correction_map_all, n, img_quat, correction_map)
    else:
        inversed_correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                            inversed_correction_map_all, n, img_quat, inversed_correction_map)

    # Rotate and update low-resolution surface coil images
    low_resolution_surface_coil_imgs = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                            low_resolution_surface_coil_imgs, n, img_quat, x2d)        

    return correction_map_all, inversed_correction_map_all, low_resolution_surface_coil_imgs



def correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, correction_map_all, n, img_quat, correction_map):
    """
    Rotates correction maps based on the specified auto-rotation mode.

    Parameters:
    - auto_rotation: Rotation mode ('Dicom' or 'LGE').
    - dim_info_org: Original dimension information.
    - data: MRI image data.
    - num_sli: Number of slices.
    - correction_map_all: All correction maps.
    - n: Current slice index.
    - img_quat: Quaternion for image orientation.
    - correction_map: Correction map to rotate.

    Returns:
    - Rotated correction map.
    """
    
    # Validate auto_rotation value
    if auto_rotation not in ['Dicom', 'LGE']:
        raise ValueError("auto_rotation should be either 'Dicom' or 'LGE'")

    # Process and rotate the correction map
    correction_map = np.abs(correction_map)
    if auto_rotation == 'Dicom':
        correction_map = np.rot90(correction_map, -1)

    try:
        if auto_rotation == 'Dicom':
            correction_map_all[n, ...] = rotate_image(correction_map, img_quat)
        else:
            correction_map_all[n, ...] = correction_map
    except Exception as e:
        #print(f"Error during rotation: {e}")
        correction_map_all = np.zeros((num_sli, data.shape[dim_info_org.index('Col')] // 2, data.shape[dim_info_org.index('Lin')]))
        correction_map_all[n, ...] = rotate_image(correction_map, img_quat) if auto_rotation == 'Dicom' else correction_map

    return correction_map_all


def sense_input_rotation(auto_rotation, img_quat, sense_data):
    """
    Rotates sense input data based on the specified auto-rotation mode.

    Parameters:
    - auto_rotation: Rotation mode ('Dicom' or 'LGE').
    - img_quat: Quaternion for image orientation.
    - sense_data: Sense data to rotate.

    Returns:
    - Rotated sense data.
    """
    #Validate auto_rotation value
    if auto_rotation not in ['Dicom', 'LGE']:
        raise ValueError("auto_rotation should be either 'Dicom' or 'LGE'")

    #Process the sense data based on the rotation mode
    if auto_rotation != 'Dicom':
        # For 'LGE' mode, reverse the last two dimensions
        return sense_data[..., ::-1, :]

    else:
        # For 'Dicom' mode, rotate and reorient the sense data
        sense_data = np.rot90(sense_data, -1, [-2, -1])
        sense_data = sense_data[:, ..., ::-1]
        sense_data_rotated = np.zeros(sense_data.shape, dtype=np.complex64)

        for ch_idx in range(sense_data.shape[0]):
            try:
                sense_data_rotated[ch_idx, ...] = rotate_image(sense_data[ch_idx, ...], img_quat)
            except Exception as e:
                print(f"Error during rotation: {e}")
                sense_data_rotated = np.zeros((sense_data.shape[0], sense_data.shape[-1], sense_data.shape[-2]), dtype=np.complex64)
                sense_data_rotated[ch_idx, ...] = rotate_image(sense_data[ch_idx, ...], img_quat)

        return sense_data_rotated



def img_normalize(uncorrected_img, percentile = 96, normalize_method = 'percentile'):
    """
    Normalizes an image by its maximum or 96th percentile values.

    Parameters:
    - uncorrected_img: The image to be normalized.

    Returns:
    - Normalized image.
    """
    corrected_img = np.abs(uncorrected_img)  # Take the absolute value
    if normalize_method == 'percentile':
        percentile_num = np.percentile(corrected_img.flatten(), percentile)
        if percentile_num != 0:
            corrected_img /= percentile_num  # Normalize by the 96th percentile
    elif normalize_method == 'max':
        if np.max(corrected_img) != 0:
            corrected_img /= np.max(corrected_img)
    else:
        raise ValueError("normalize_method should be either 'percentile' or 'max'")

    return corrected_img

import matplotlib.gridspec as gridspec


def displaying_results(base_dir, input_folder, output_folder, folder_names=None, sli_idx=0, par_idx=None, 
                       avg_idx=None, fig_h=9, show_both=False):
    """
    Displays MRI scan results including correction maps, and uncorrected/corrected images.

    Parameters:
    - base_dir: Base directory for input and output folders.
    - input_folder: Name of the input folder.
    - output_folder: Name of the output folder.
    - folder_names: List of folder names to process. If None, all folders are processed.
    - sli_idx: Slice index to display.
    - par_idx: Parallel imaging parameter index.
    - avg_idx: Average index (unused in current implementation).
    - fig_h: Figure height for the plots.
    - show_both: Flag to indicate whether to show both image corrected and sense corrected images.
    """
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
