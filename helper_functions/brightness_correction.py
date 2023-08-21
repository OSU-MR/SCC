import os
import numpy as np
from helper_functions.preprocess import ifftnd, rms_comb,remove_RO_oversamling
from helper_functions.read_data import readtwix_arry_all
from helper_functions.Interpolation import generate_3D_data, interpolation, quaternion_to_directions
from helper_functions.calculating_correction_map import calculate_correction_map, normalize_images
#from brightness_correction.recon import grappa_reconstruction
from helper_functions.recon import sense_reconstruction, remove_edges, rotate_image, pad_ref
from matplotlib import pyplot as plt

#######################################
def defaultProcedure(img_body_coils, img_surface_coils, ksp, ref_padded, noise_kspace, dim_info_noise):

    img_body_coils = rms_comb(img_body_coils,0)
    img_surface_coils = rms_comb(img_surface_coils,0)

    return img_body_coils, img_surface_coils, ksp, ref_padded

#######################################

def rotate_images_for_LGE(results, correction_map_all,inversed_correction_map_all, quat, filename):

    read_dir , phase_dir, _ = quaternion_to_directions(quat)

    if filename.find('LAX') != -1:
        if read_dir[0] > 0:
            #flip the image
            results = np.flip(results, axis = 2)
            correction_map_all = np.flip(correction_map_all, axis = 2)
            inversed_correction_map_all = np.flip(inversed_correction_map_all, axis = 2)
        #rotate the image clockwise 90 degree
        results = np.rot90(results,k=-1, axes = (1,2))
        correction_map_all = np.rot90(correction_map_all, k=-1,axes = (1,2))
        inversed_correction_map_all = np.rot90(inversed_correction_map_all, k=-1,axes = (1,2))

    if filename.find('SAX') != -1:
        if read_dir[-1] > 0:
            #flip the image
            results = np.flip(results, axis = 2)
            correction_map_all = np.flip(correction_map_all, axis = 2)
            inversed_correction_map_all = np.flip(inversed_correction_map_all, axis = 2)
        if phase_dir[0] < 0:
            #flip the image
            results = np.flip(results, axis = 1)
            correction_map_all = np.flip(correction_map_all, axis = 1)
            inversed_correction_map_all = np.flip(inversed_correction_map_all, axis = 1)
        #rotate the image
        results = np.rot90(results, axes = (1,2))
        correction_map_all = np.rot90(correction_map_all, axes = (1,2))
        inversed_correction_map_all = np.rot90(inversed_correction_map_all, axes = (1,2))

    if filename.find('2CH') != -1:
        if read_dir[-1] > 0:
            #flip the image
            results = np.flip(results, axis = 2)
            correction_map_all = np.flip(correction_map_all, axis = 2)
            inversed_correction_map_all = np.flip(inversed_correction_map_all, axis = 2)
        if phase_dir[0] < 0 and phase_dir[1] > 0:
            #flip the image
            results = np.flip(results, axis = 1)
            correction_map_all = np.flip(correction_map_all, axis = 1)
            inversed_correction_map_all = np.flip(inversed_correction_map_all, axis = 1)
        #rotate the image
        results = np.rot90(results, axes = (1,2))
        correction_map_all = np.rot90(correction_map_all, axes = (1,2))
        inversed_correction_map_all = np.rot90(inversed_correction_map_all, axes = (1,2))

    return results, correction_map_all ,inversed_correction_map_all



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

def middle_slice(data,data_dimensions, dims_to_keep = ['Sli', 'Lin', 'Cha', 'Col'], dim_to_set_to_zero = ['Phs','Set']):
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

def brightness_correction_map_generator(data_path, filename_matched, auto_rotation = 'Dicom', apply_correction_during_sense_recon = False, CustomProcedure = None):
    
    #twix, data_org, dim_info_org,data_ref, dim_info_ref, noise_kspace, dim_info_noise = readtwix_arry_all(data_path, filename_matched)
    twix, mapped_data,data_org, dim_info_org ,data_ref, dim_info_ref, noise_kspace, dim_info_noise = readtwix_arry_all(data_path, filename_matched)
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


    data,dim_info_org = middle_slice(data,data_dimensions = dim_info_org, dims_to_keep = ['Sli', 'Lin', 'Cha', 'Col'])

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
                correction_map_all, ksp, ref_padded, img_quat, normal, x2d, inversed_correction_map_all = calculating_correction_maps(auto_rotation, 
                                                                                                                                    CustomProcedure, 
                                                                                                                                    twix, dim_info_org, 
                                                                                                                                    data, image_3D_body_coils, 
                                                                                                                                    image_3D_surface_coils, 
                                                                                                                                    num_sli, correction_map_all,
                                                                                                                                    inversed_correction_map_all,
                                                                                                                                    n,ksp,ref_padded,noise_kspace, dim_info_noise)
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
            correction_map_all, ksp, ref_padded, img_quat, normal, x2d, inversed_correction_map_all = calculating_correction_maps(auto_rotation, 
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
    return recon_results,correction_map_all,img_quat,normal, inversed_correction_map_all, apply_correction_during_sense_recon

def calculating_correction_maps(auto_rotation, CustomProcedure, twix, dim_info_org, 
                                data, image_3D_body_coils, image_3D_surface_coils, 
                                num_sli, correction_map_all,inversed_correction_map_all, n,
                                ksp = None , ref_padded = None , noise_kspace = None , dim_info_noise = None ):
    Zi_body_coils, Zi_surface_coils, img_quat , normal= interpolation(twix, image_3D_body_coils, image_3D_surface_coils, num_sli, n)
    inter_img_body_coils, inter_img_surface_coils = remove_edges(Zi_body_coils,Zi_surface_coils)
    inter_img_body_coils = inter_img_body_coils.transpose([2,0,1])
    inter_img_surface_coils = inter_img_surface_coils.transpose([2,0,1])

    #see if these is a function called CustomProcedure else use defaultProcedure
    try:
        inter_img_body_coils, inter_img_surface_coils, ksp, ref_padded = CustomProcedure(inter_img_body_coils,
                                                                                                         inter_img_surface_coils, 
                                                                                                         ksp, ref_padded, noise_kspace, dim_info_noise)
        #print with green color
        print("\033[92m" + "custom procedure found and used!" + "\033[0m")
    except Exception as e:
        if CustomProcedure != None:
            #print with red color
            print("\033[91m" + "custom procedure found, but there is something wrong with it. Use default procedure!" + "\033[0m")
            print(e)
        else:
            #print with yellow color
            print("\033[93m" + "no custom procedure found, use default procedure!" + "\033[0m")
        inter_img_body_coils, inter_img_surface_coils, ksp, ref_padded = defaultProcedure(inter_img_body_coils,
                                                                                                          inter_img_surface_coils, 
                                                                                                          ksp, ref_padded, noise_kspace, dim_info_noise)
        

    x2d, x3d = normalize_images(inter_img_surface_coils,inter_img_body_coils)#,scanned_img)


    *_ ,correction_map = calculate_correction_map(A=x2d,B=x3d,lamb=1e-1)
    *_ ,inversed_correction_map = calculate_correction_map(A=x3d,B=x2d,lamb=1e-1)

    correction_map_all, inversed_correction_map_all = correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, 
                                                                                        correction_map_all, inversed_correction_map_all, n, 
                                                                                        img_quat, correction_map, inversed_correction_map)
                                                                        
    return correction_map_all,ksp,ref_padded,img_quat,normal,x2d,inversed_correction_map_all

def correction_map_rotation(auto_rotation, dim_info_org, data, num_sli, correction_map_all
                            , inversed_correction_map_all, n, img_quat, correction_map, inversed_correction_map):
    if auto_rotation == 'Dicom':
        correction_map = np.rot90(correction_map,-1)
        inversed_correction_map = np.rot90(inversed_correction_map,-1)

    try:
        if auto_rotation == 'Dicom':
            correction_map_all[n,...] = rotate_image(correction_map,img_quat)
            inversed_correction_map_all[n,...] = rotate_image(inversed_correction_map,img_quat)
        else:
            correction_map_all[n,...] = correction_map
            inversed_correction_map_all[n,...] = inversed_correction_map
            
    except:
        if auto_rotation == 'Dicom':
            correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
            correction_map_all[n,...] = rotate_image(correction_map,img_quat)
            inversed_correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
            inversed_correction_map_all[n,...] = rotate_image(inversed_correction_map,img_quat)
        else:
            correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
            correction_map_all[n,...] = correction_map
            inversed_correction_map_all = np.zeros((num_sli,data.shape[dim_info_org.index('Col')]//2,data.shape[dim_info_org.index('Lin')]))
            inversed_correction_map_all[n,...] = inversed_correction_map
    return correction_map_all,inversed_correction_map_all

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
                    recon_results, correction_map_all, quat, _, inversed_correction_map_all, apply_correction_during_sense_recon = brightness_correction_map_generator(full_dir_name, filename, 
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

def displaying_results(base_dir ,input_folder, output_folder, folder_names = None, sli_idx = 0, avg_idx = None, fig_h = 9, debug = False):
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
                
                    correction_map_all_filename = os.path.join(full_dir_name_output, f"{filename}.image_correction_map.npy")
                    correction_map_all = np.load(correction_map_all_filename)
                    inversed_correction_map_all_filename = os.path.join(full_dir_name_output, f"{filename}.sensitivity_correction_map.npy")
                    inversed_correction_map_all = np.load(inversed_correction_map_all_filename)

                    try:
                        corrected_results_filename = os.path.join(full_dir_name_output, f"{filename}.corrected_sense_results.npy")
                        corrected_results = np.load(corrected_results_filename)
                    except:
                        corrected_results = None
                    try:
                        uncorrected_results_filename = os.path.join(full_dir_name_output, f"{filename}.uncorrected_results.npy")
                        uncorrected_results = np.load(uncorrected_results_filename)
                    except:
                        uncorrected_results = None

                    try:
                        quat_filename = os.path.join(full_dir_name_output, f"{filename}.quat.npy")
                        quat = np.load(quat_filename)
                        *_, slc_dir_vec = quaternion_to_directions(quat)

                    except:
                        ...


                    img_shape = correction_map_all.shape[-2:]
                    if img_shape[0] > img_shape[1]:
                        auto_width = int(( np.max(img_shape) / np.min(img_shape) )*fig_h)
                    else:
                        auto_width = 18


                    if debug:

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
                                delta_for_colorbar = 9
                                # Create a GridSpec object
                                gs = gridspec.GridSpec(subdevide_num, ncols=ncols, width_ratios=width_ratios, figure=plt.figure(figsize=(auto_width,fig_h))) # Adjust the figure size as needed

                                # Create subplots and colorbar axes
                                cax1 = plt.subplot(gs[delta_for_colorbar:subdevide_num-delta_for_colorbar, 0])
                                ax1 = plt.subplot(gs[0:subdevide_num, 1])
                                ax2 = plt.subplot(gs[0:subdevide_num, 2])

                                im1 = ax1.imshow(np.abs(correction_map_all[sli_idx,...]),cmap='jet')
                                ax1.axis("off")
                                ax1.set_title("image correction map")

                                im2 = ax2.imshow(np.abs(inversed_correction_map_all[sli_idx,...]),cmap='jet')
                                ax2.axis("off")
                                ax2.set_title("sensetivity correction map")

                                if uncorrected_results is not None:

                                    ax3 = plt.subplot(gs[0:subdevide_num, 3])
                                    ax4 = plt.subplot(gs[0:subdevide_num, 4])

                                    uncorrected_img = img_normalize(uncorrected_results[sli_idx,...])
                                    im3 = ax3.imshow(uncorrected_img**1,vmax=1,cmap='gray')
                                    ax3.axis("off")
                                    ax3.set_title("before correction")

                                    corrected_img = np.multiply(correction_map_all[sli_idx,...],uncorrected_results[sli_idx,...])
                                    corrected_img = img_normalize(corrected_img)
                                    im4 = ax4.imshow(corrected_img**1,cmap='gray',vmax=1)
                                    ax4.axis("off")
                                    ax4.set_title("uncorrected results ⊙ correction map")

                                    if corrected_results is not None:
                                        ax5 = plt.subplot(gs[0:subdevide_num, 5])
                                        corrected_img = img_normalize(corrected_results[sli_idx,...])
                                        im5 = ax5.imshow(corrected_img**1,cmap='gray',vmax=1)
                                        ax5.axis("off")
                                        ax5.set_title("corrected results\n(generated during\nsense reconstruction)")
                                elif corrected_results is not None:
                                    ax3 = plt.subplot(gs[0:subdevide_num, 3])
                                    corrected_img = img_normalize(corrected_results[sli_idx,...])
                                    im3 = ax3.imshow(corrected_img**1,cmap='gray',vmax=1)
                                    ax3.axis("off")
                                    ax3.set_title("corrected results\n(generated during\nsense reconstruction)")

                            
                                # Display colorbar for the first subplot
                                plt.colorbar(im1, cax=cax1)
                                try:
                                    plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + '\n' +"quat: "+str(quat)+"\n"+"slice vector: "+str(slc_dir_vec)+"\n\n", x=0.5, y=0.80, ha='center')
                                except:
                                    plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + "\n\n", x=0.5, y=0.80, ha='center')

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
                                    plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename + '\n' +"quat: "+str(quat)+"\n"+"slice vector: "+str(slc_dir_vec))
                                except:
                                    plt.suptitle("path to file: " + full_dir_name_output + "\n" + "file name: " + filename)

                                plt.tight_layout();plt.show()
                            except Exception as e:
                                print(e)
                                ...