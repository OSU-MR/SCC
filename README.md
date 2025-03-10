[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OSU-MR/SCC/blob/main/brightness_correction_demo.ipynb)



## Surface Coil Intensity Correction (SCC)


1. Create a new environment named SCC in Conda and activate it with the following command (Don't close the terminal, we will use it in step 3):
   * Open a new terminal and run the following command:
   ```
   conda create --name SCC python jupyterlab ipykernel -y && conda activate SCC && python -m ipykernel install --user --name=SCC --display-name "SCC"
   ```

> [!IMPORTANT]
> If you have **compatibility issues**, try the following command to create your Conda environment instead:
>   ```
>   conda create --name SCC python=3.8 jupyterlab ipykernel -y && conda activate SCC && python -m ipykernel install --user --name=SCC --display-name "SCC"
>   ```

2. Clone the GitHub repository to your local workstation

3. Open `brightness_correction_demo.ipynb` in Anaconda, VS Code, etc.

   **In the terminal you opened in step 1, do the following steps:**
   * Navigate to the folder you just downloaded using:  `cd path_to_the_folder_you_just_downloaded`
   * Run `jupyter lab` in the terminal
   * Execute all cells in `brightness_correction_demo.ipynb` for a quick verification. If you can see the demo output, it means you have successfully installed and configured SCC.
  >   [!important]  
  >   Ensure the SCC environment is selected before executing the cells!

  >   [!TIP]  
  >   The code works with Siemens rawdata (.dat format).

   


## For installing the package content (optional)
Navigate to the path of the SCC folder and run the following command:
```
pip install -e
```

## Representative results
<p align="center">
  <img src="2ch-results.png" alt="representative results"/>
  <br/>
  <i>Figure 1: From left to right, correction map for the image, correction map for the sensitivity maps, magnitude of the uncorrected image, magnitude of the image corrected with the first correction map, the magnitude of the image where the sensitivity maps are corrected with the second correction map. A representative two-chamber view of the heart is shown.</i>
  <img src="LAX-results.png" alt="representative results"/>
  <br/>
  <i>Figure 2: From left to right, correction map for the image, correction map for the sensitivity maps, magnitude of the uncorrected image, magnitude of the image corrected with the first correction map, the magnitude of the image where the sensitivity maps are corrected with the second correction map. A representative long axis view of the heart is shown.</i>
</p>

## Pulication
[eprint arXiv:2312.00936](https://arxiv.org/abs/2312.00936)

   


