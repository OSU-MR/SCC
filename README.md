# Open-Source_Siemens_Under-lit_Mitigation_Corrector
alpha version of the raw dataset corrector

Please download and install the latest release of our Python package.

Make sure you have installed twixtools modified by Chong!!!
Here's the link:
https://github.com/OSU-MR/Python_read_Siemens_rawdata


1.Install The Package:

pip3 install brightness_correction-0.x.tar.gz

2.Install twixtools modified by Chong

https://github.com/OSU-MR/Python_read_Siemens_rawdata

3.Import The Functions And Set The Path Of Your Datasets:

The sturcture of your dataset folders should be:


	base_dir-----input_folder-----folder name of your datasets_1
	          |                |--folder name of your datasets_2
	          |                |              ...
	          |		   --folder name of your datasets_n
	          |
	          |
	          |
	          ---output_folder------correction map folder of your datasets_1(these folders in the output folder 
				     |--correction map folder of your datasets_2      will be automatically created)
				     |                ...                 
	                             ---correction map folder of your datasets_n



	from brightness_correction.brightness_correction import getting_correction_map, create_and_start_threadings, displaying_results

	base_dir = "/your_project"
	input_folder = "rawdata"
	output_folder = "correction_map"
	folder_names = ['folder name of your datasets1'] #if this variable is None then the algorithm 		
	                                                  generates correction map for all datasets inside input folder

4.Run The Code
number_of_threads = 1 # how many threads you want to run at the same time

threads = create_and_start_threadings(number_of_threads, getting_correction_map, base_dir, input_folder, output_folder, folder_names, auto_rotation='LGE',debug = True)


You can check the if the threads are alive with:

	 for i in range(len(threads)):
	    print(threads[i].is_alive())


5.Check The Reault Visually:

This code displays the result images along with the debug parameters

	folder_names = ['folder name of your datasets1']
	displaying_results(base_dir=base_dir, input_folder=input_folder,
	                   output_folder=output_folder, folder_names=folder_names, 
		           sli_idx=0, avg_idx=None, fig_h=5)
