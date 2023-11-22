import numpy as np
from helper_functions.preprocess import adjust_rawdata_dimmension, ifftnd, rms_comb
from scipy.interpolate import RegularGridInterpolator

def generate_3D_data(mapped_data ,pre_scan = 0):
    #we use the first scan as the pre scan, if it is not the pre scan, please change it

    im_data = mapped_data[pre_scan]['image']

    # the twix_array object makes it easy to remove the 2x oversampling in read direction
    im_data.flags['remove_os'] = False  # not handle assymetric echo
    im_data.flags['zf_missing_lines'] = True
    im_data.flags['average']['Seg'] = True
    im_data.flags['average']['Ave'] = False

    # read the data (array-slicing is also supported)
    data_org = im_data[:].squeeze()
    dim_info_org_pre = im_data.non_singleton_dims


    ## reshape the data
    # set-0: data from small coils
    # set-1: data from large coils (only first 2 channels are valid)
    print('original data shape:', data_org.shape)
    print(dim_info_org_pre)
    # rearrange the data array
    data_pre, dim_info = adjust_rawdata_dimmension(data_org, dim_info_org_pre)
    print('arranged data shape', data_pre.shape)
    print(dim_info)  

    image_3D = np.squeeze(ifftnd(data_pre, [0,1,2]))
    print('3d image shape:', image_3D.shape)

    tmp_image_3D_body_coils = image_3D[:,:,:,0:2,1]#(np.abs(image_3D[:,:,:,0,1])**2 + np.abs(image_3D[:,:,:,1,1])**2)**0.5
    tmp_image_3D_surface_coils = image_3D[:,:,:,:,0]#rms_comb(image_3D[:,:,:,:,0], axis=3)
    return tmp_image_3D_body_coils , tmp_image_3D_surface_coils


def rps_from_quat(data_idx = 0, img_ori = None, twix= None):

    try:
        image_mdbs = [mdb for mdb in twix[data_idx]['mdb'] if mdb.is_image_scan()]
        img_ori = image_mdbs[0].mdh.SliceData.Quaternion if img_ori is None else img_ori
        #print(img_ori)
    except:
        image_mdbs = [mdb for mdb in twix[data_idx]['mdb'] if mdb['is_image_scan']]
        img_ori = image_mdbs[0]['Quaternion'] if img_ori is None else img_ori

    
    read_dir, phase_dir, slice_dir = quaternion_to_directions(img_ori)

    #print(np.round(read_dir,5),'\n', np.round(phase_dir,5),'\n', np.round(slice_dir,5))
    return np.array([read_dir, phase_dir, slice_dir])


def quaternion_to_directions(quat):

# a = quat(1); b = quat(2); c = quat(3); d = quat(4);

# assume x = quat(2), y = quat(3), z = quat(4), w = quat(1).

# then
    read_dir  = np.zeros(3)
    phase_dir = np.zeros(3)
    slice_dir = np.zeros(3)

    a = quat[1]; b = quat[2]; c = quat[3]; d = quat[0]

    phase_dir[0]  = 1.0 - 2.0 * ( b*b + c*c )
    read_dir[0] =       2.0 * ( a*b - c*d )
    slice_dir[0] =       2.0 * ( a*c + b*d )

    phase_dir[1]  =       2.0 * ( a*b + c*d )
    read_dir[1] = 1.0 - 2.0 * ( a*a + c*c )
    slice_dir[1] =       2.0 * ( b*c - a*d )

    phase_dir[2]  =       2.0 * ( a*c - b*d )
    read_dir[2] =       2.0 * ( b*c + a*d )
    slice_dir[2] = 1.0 - 2.0 * ( a*a + b*b )

    return read_dir, phase_dir, slice_dir


def points_rps2xyz(scan_index = 0, twix = None, 
                   fov = None, resolution = None, 
                   rotmatrix = None, offset = None,
                   rotmatrix_3d = None ,offset_0 = None,
                   voxelsize = None, num_sli = None, n = None, oversampling_phase_factor = 3):
    


    #print("scan_index",scan_index)
    reado_idx = 0
    phase_idx = 1
    slice_idx = 2

    geo = twix[scan_index]['geometry']
    resolution = geo.resolution if resolution is None else resolution
    fov = geo.fov if fov is None else fov
    voxelsize = geo.voxelsize if voxelsize is None else voxelsize

    ##################
    rotmatrix = np.array(geo.rotmatrix) if rotmatrix is None else np.array(rotmatrix)
    ###################
    try:
        image_mdbs = [mdb for mdb in twix[scan_index]['mdb'] if mdb.is_image_scan()]
        img_quat = image_mdbs[(len(image_mdbs)//num_sli)*n].mdh.SliceData.Quaternion
    except:
        image_mdbs = [mdb for mdb in twix[scan_index]['mdb'] if mdb['is_image_scan']]
        img_quat = image_mdbs[(len(image_mdbs)//num_sli)*n]['Quaternion']

    if scan_index != 0:
        try:
            offset  = image_mdbs[(len(image_mdbs)//num_sli)*n].mdh.SliceData.SlicePos
        except:
            offset  = image_mdbs[(len(image_mdbs)//num_sli)*n]['SlicePos']
        offset  = [offset.Sag,offset.Cor,offset.Tra]
    #print(img_quat)
    #print(offset)
    ##############
    GlobalTablePosCor = 0 if twix[scan_index]['hdr']['Config']['GlobalTablePosCor'] == '' else twix[scan_index]['hdr']['Config']['GlobalTablePosCor']
    GlobalTablePosSag = 0 if twix[scan_index]['hdr']['Config']['GlobalTablePosSag'] == '' else twix[scan_index]['hdr']['Config']['GlobalTablePosSag']
    GlobalTablePosTra = 0 if twix[scan_index]['hdr']['Config']['GlobalTablePosTra'] == '' else twix[scan_index]['hdr']['Config']['GlobalTablePosTra']
    #print("table position",GlobalTablePosCor,GlobalTablePosSag,GlobalTablePosTra)

    offset = np.array(geo.offset) if offset is None else np.array(offset)
    offset = np.array([-offset[0] , -offset[1]  ,-offset[2] ])
    #print("offset",offset)

    r = 0.5*np.linspace(-1,1,resolution[reado_idx])*fov[reado_idx]+voxelsize[reado_idx]/2
    p = 0.5*np.linspace(-1,1,resolution[phase_idx])*fov[phase_idx]+voxelsize[phase_idx]/2
    s = 0.5*np.linspace(-1,1,resolution[slice_idx])*fov[slice_idx]-voxelsize[slice_idx]/2
    
    if scan_index == 1:
        ...
        #start_point = 0
        #s = np.linspace(start_point,start_point+1,resolution[slice_idx])*fov[slice_idx]
        #s = 0.5*np.linspace(-1,1,resolution[slice_idx]*1)*fov[slice_idx]+voxelsize[slice_idx]/2
        p = 0.5*np.linspace(-1,1,resolution[phase_idx]*oversampling_phase_factor)*fov[phase_idx]*oversampling_phase_factor+voxelsize[phase_idx]/2
    

    [RR,PP,SS] = np.meshgrid(r[::-1], p[::-1], s[::-1])

    points_rps = np.vstack([RR.ravel(), PP.ravel(), SS.ravel()])#.T

    if scan_index == 0:
        #print("rotmatrix:\n",rotmatrix)
        points_xyz = points_rps
    elif scan_index == 1:
 
        points_xyz = np.dot( rps_from_quat(0,twix = twix) , (np.dot(  np.linalg.inv(rps_from_quat(1,twix = twix))  ,points_rps) + offset[:, np.newaxis]) -   offset_0[:, np.newaxis] )

    #print(geo)
    #print(geo.resolution)
    #print(geo.fov)
    
    return points_xyz.reshape((-1,)+RR.shape) , rotmatrix, offset , img_quat, geo.normal

import matplotlib.pyplot as plt
def interpolation(twix, data_org, dim_info_org, num_sli, n, input_data, oversampling_phase_factor = 3):
    points_3d_xyz , rotmatrix3d, offset, _,_ = points_rps2xyz(0,twix= twix,num_sli = num_sli, n = n)

    points_2d_xyz , *_ , img_quat, normal = points_rps2xyz(1, twix= twix,rotmatrix_3d = rotmatrix3d, offset_0 = offset,num_sli = num_sli, n = n, oversampling_phase_factor = oversampling_phase_factor)
    points_2d_xyz = points_2d_xyz.transpose([1,2,3,0])#.reshape((-1,3))
    #when dim_info_org.index('Par') has a value and the value is bigger than 1 we don't need points_2d_xyz = np.mean(points_2d_xyz,2), the 'Par' doesn't always exist in the dim_info_org
    # print(dim_info_org)
    # print("points_2d_xyz",points_2d_xyz.shape)
    # print("data_org",data_org.shape)
    
    try:
        #print("data_org.shape[dim_info_org.index('Par')]",data_org.shape[dim_info_org.index('Par')])
        if data_org.shape[dim_info_org.index('Par')] <= 1:
            points_2d_xyz = np.mean(points_2d_xyz,2)
    except:
        points_2d_xyz = np.mean(points_2d_xyz,2)
    #print("points_2d_xyz",points_2d_xyz.shape)

    output_data = []
    for data in input_data:
        if data is not None:
            output_data.append(cut_3D_cube(data, points_3d_xyz, points_2d_xyz))
        else:
            output_data.append(None)

    return output_data, img_quat, normal


def cut_3D_cube(tmp_image_3D, points_3d_xyz, points_2d_xyz):
    try:
        dummy_var = tmp_image_3D.shape[3]
    except:
        tmp_image_3D = np.expand_dims(tmp_image_3D, axis=3)

    if len(points_2d_xyz.shape) == 3:
        interpolated_img = np.zeros((points_2d_xyz.shape[0],points_2d_xyz.shape[1],tmp_image_3D.shape[3]))
    elif len(points_2d_xyz.shape) == 4:
        interpolated_img = np.zeros((points_2d_xyz.shape[0],points_2d_xyz.shape[1],points_2d_xyz.shape[2],tmp_image_3D.shape[3]))
    else:
        raise ValueError("unknown data shape")

    #print("points_2d_xyz.shape",points_2d_xyz.shape) #(96, 192, 72, 3)
    #for par ==1, points_2d_xyz (150, 512, 3)

    for coil_idx in range(tmp_image_3D.shape[3]):
        interpolator = RegularGridInterpolator((points_3d_xyz[0, 0, :, 0],
                                        points_3d_xyz[1, :, 0, 0],
                                        points_3d_xyz[2, 0, 0, :]),
                                        abs(tmp_image_3D[:,:,::-1,coil_idx]),  bounds_error=False,fill_value = 0) #[:,:,::-1]
        zi_image = interpolator(points_2d_xyz[...,[0,1,2]])
        #np.save("correction_map.npy",zi_image)
        interpolated_img[...,coil_idx] = zi_image.reshape(points_2d_xyz.shape[:-1])
    return np.squeeze(interpolated_img)


# def cut_3D_cube(tmp_image_3D, points_3d_xyz, points_2d_xyz):
#     try:
#         dummy_var = tmp_image_3D.shape[3]
#     except:
#         tmp_image_3D = np.expand_dims(tmp_image_3D, axis=3)

#     interpolated_2D_img = np.zeros((points_2d_xyz.shape[0],points_2d_xyz.shape[1],tmp_image_3D.shape[3]))
#     for coil_idx in range(tmp_image_3D.shape[3]):
#         interpolator = RegularGridInterpolator((points_3d_xyz[0, 0, :, 0],
#                                         points_3d_xyz[1, :, 0, 0],
#                                         points_3d_xyz[2, 0, 0, :]),
#                                         abs(tmp_image_3D[:,:,::-1,coil_idx]),  bounds_error=False,fill_value = 0)#np.nan) #[:,:,::-1]
#         zi_image = interpolator(points_2d_xyz[...,[0,1,2]])
#         interpolated_2D_img[:,:,coil_idx] = zi_image.reshape(points_2d_xyz.shape[:2])
#     return np.squeeze(interpolated_2D_img)