import os
import gzip
import pickle
from helper_functions.brightness_correction import rawdata_reader

def change_file_extension(filename, new_extension):
    base_name = os.path.splitext(filename)[0]
    return base_name + new_extension

def extract_fields(entry):
    new_data = {}

    # Always keeping these
    # Transform the 'mdb' data to retain only the necessary information
    if 'mdb' in entry:
        new_data['mdb'] = []
        for mdb_obj in entry['mdb']:
            mdb_dict = {
                "is_image_scan": mdb_obj.is_image_scan(),
                "Quaternion": mdb_obj.mdh.SliceData.Quaternion if mdb_obj.is_image_scan() else None,
                #"SlicePos" : [mdb_obj.mdh.SliceData.SlicePos.Sag, mdb_obj.mdh.SliceData.SlicePos.Cor, mdb_obj.mdh.SliceData.SlicePos.Tra] if mdb_obj.is_image_scan() else None
                "SlicePos" : mdb_obj.mdh.SliceData.SlicePos if mdb_obj.is_image_scan() else None
            }
            new_data['mdb'].append(mdb_dict)

            
    new_data['geometry'] = entry['geometry']

    # Ensure 'hdr' exists in new_data
    if 'hdr' in entry:
        new_data['hdr'] = {}

        # Extracting 'Config' fields
        if 'Config' in entry['hdr']:
            new_data['hdr']['Config'] = {}
            config_fields = ['GlobalTablePosCor', 'GlobalTablePosSag', 'GlobalTablePosTra']
            for field in config_fields:
                if field in entry['hdr']['Config']:
                    new_data['hdr']['Config'][field] = entry['hdr']['Config'][field]

        # Ensure 'MeasYaps' exists in both entry and new_data
        if 'MeasYaps' in entry['hdr']:
            new_data['hdr']['MeasYaps'] = {}

            # Extracting sKSpace fields
            if 'sKSpace' in entry['hdr']['MeasYaps']:
                new_data['hdr']['MeasYaps']['sKSpace'] = {}
                sKSpace_fields = ['ucDimension', 'lBaseResolution', 'lPhaseEncodingLines', 'lPartitions']
                for field in sKSpace_fields:
                    if field in entry['hdr']['MeasYaps']['sKSpace']:
                        new_data['hdr']['MeasYaps']['sKSpace'][field] = entry['hdr']['MeasYaps']['sKSpace'][field]

            # Extracting sSliceArray fields
            if 'sSliceArray' in entry['hdr']['MeasYaps'] and 'asSlice' in entry['hdr']['MeasYaps']['sSliceArray']:
                new_data['hdr']['MeasYaps']['sSliceArray'] = {}
                new_data['hdr']['MeasYaps']['sSliceArray']['asSlice'] = entry['hdr']['MeasYaps']['sSliceArray']['asSlice']

        # Extracting Meas fields
        if 'Meas' in entry['hdr']:
            new_data['hdr']['Meas'] = {}
            if 'tPatientPosition' in entry['hdr']['Meas']:
                new_data['hdr']['Meas']['tPatientPosition'] = entry['hdr']['Meas']['tPatientPosition']
            if 'sPatPosition' in entry['hdr']['Meas']:
                new_data['hdr']['Meas']['sPatPosition'] = entry['hdr']['Meas']['sPatPosition']

    return new_data

def data_publishing(path_input):
    '''
    you can use this function to remove the sensitive information of the rawdata for publishing
    
    '''
    for i, data_path_filename in enumerate(path_input):
    # This is assuming you read the twix and other variables from rawdata_reader function
        twix, image_3D_body_coils, image_3D_surface_coils, data, dim_info_data, data_ref, dim_info_ref, num_sli = rawdata_reader(data_path_filename)

    # Transform the twix data using extract_fields function
        twix_transformed = [extract_fields(entry) for entry in twix]

    # Pack all the variables into a dictionary
        packed_data = {
        'twix': twix_transformed,
        'image_3D_body_coils': image_3D_body_coils,
        'image_3D_surface_coils': image_3D_surface_coils,
        'data': data,
        'dim_info_data': dim_info_data,
        'data_ref': data_ref,
        'dim_info_ref': dim_info_ref,
        'num_sli': num_sli
    }

    # Save the packed data to a file with .demo extension, using gzip compression
        demo_filename = change_file_extension(data_path_filename, ".demo.gz")  # Note the added .gz extension
        with gzip.open(demo_filename, 'wb') as f:
            pickle.dump(packed_data, f)
        print("Saved simplified data to", demo_filename)