#code for reading rawdata
import twixtools
import numpy as np
import os

def readtwix_arry_all(data_path = None, filename = None, data_path_filename = None):
    '''
    given a rawdata file, read all the nessacery data from the file
    args:
        data_path: path of the rawdata
        filename: name of the rawdata
    '''
    if data_path_filename is not None:
        try:
            twix = twixtools.read_twix(data_path_filename, keep_syncdata_and_acqend=True)
        except:
            try:
                twix = twixtools.read_twix(data_path_filename, parse_pmu = False, keep_syncdata = True , keep_acqend = True)
            except:
                twix = twixtools.read_twix(data_path_filename, parse_pmu = False, keep_syncdata = False , keep_acqend = True)
    elif data_path is not None and filename is not None:
        try:
            twix = twixtools.read_twix(os.path.join(data_path, filename), keep_syncdata_and_acqend=True)
        except:
            try:
                twix = twixtools.read_twix(os.path.join(data_path, filename), parse_pmu = False, keep_syncdata = True , keep_acqend = True)
            except:
                twix = twixtools.read_twix(os.path.join(data_path, filename), parse_pmu = False, keep_syncdata = False , keep_acqend = True)
    #mapped = twixtools.map_twix(twix[-1])
    print('\nnumber of separate scans (multi-raid):', len(twix))
    
    # map the twix data to twix_array objects
    mapped = twixtools.map_twix(twix)
    data_org, dim_info_org = read_specific_block("image", twix, mapped)
    try:
        data_ref, dim_info_ref = read_specific_block("refscan", twix, mapped)
    except:
        data_ref = None
        dim_info_ref = None
    
    #block for read noise, if you need this part, please uncomment the code below
    #sort all 'noise' mdbs into a k-space array
    noise_mdbs = [mdb for mdb in twix[-1]['mdb'] if (mdb.get_active_flags()[-1] == 'NOISEADJSCAN')]
    noise_kspace = np.array([])
    dim_info_noise = []
    if len(noise_mdbs) > 0:
        # assume that all data were acquired with same number of channels & columns:
        n_channel, n_column = noise_mdbs[0].data.shape
        noise_kspace = np.zeros([len(noise_mdbs), n_channel, n_column], dtype=np.complex64)
        idx = 0
        for mdb in noise_mdbs:   
            noise_kspace[idx] = mdb.data
            idx += 1
        noise_kspace = np.moveaxis(noise_kspace, 0, 2)
        dim_info_noise = ['Cha', 'Col', 'lines']
    
    return twix,mapped, data_org, dim_info_org ,data_ref, dim_info_ref, noise_kspace, dim_info_noise

def read_specific_block(data_name, twix, mapped, index_for_scan = -1):
    '''
    args:
        data_name: 'image' or 'refscan'
        twix: twix object
        mapped: mapped twix object
        index_for_scan: the index of scan, default is the last scan
    '''
    #we use the last scan as default, if it is not the actual scan, please change the index_for_scan
    im_data = mapped[index_for_scan][data_name]

    # the twix_array object makes it easy to remove the 2x oversampling in read direction
    im_data.flags['remove_os'] = False #not handle assymetric echo
    im_data.flags['zf_missing_lines'] = True
    im_data.flags['average']['Seg'] = True
    im_data.flags['average']['Ave'] = False

    # read the data (array-slicing is also supported)
    data = im_data[:].squeeze()
    dim_info = im_data.non_singleton_dims
    
    # deal with the assymetric echo
    for mdb in twix[-1]['mdb']:
        if mdb.is_image_scan():
            tmp_mdb = mdb
            break
    pre_z = tmp_mdb.data.shape[1] - 2*tmp_mdb.mdh.CenterCol # size of pre padding
    num_dim = len(data.shape)
    padsize = [(0,0)]*num_dim
    dim_E0 = dim_info.index('Col')
    padsize[dim_E0] = (pre_z, 0)
    #print('pad size:',padsize)
    #print(dim_info)
    data = np.pad(data, padsize,'constant', constant_values=(0,0))
    return data,dim_info
