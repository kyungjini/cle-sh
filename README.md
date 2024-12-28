# CLE-Sh
This repository contains the code and datasets for the paper “CLE-SH: Comprehensive Literal Explanation Package for SHapley Values by Statistical Validity.”

The repository includes the CLE-SH source code as well as example testing scripts used to generate results illustrated in the paper.

To reproduce the results presented in the paper, download the datasets, place them in the appropriate directories, and execute the provided script files.

## Overview of Code Files
Below is a description of the key files in the repository:

1. **learning.py**

This script calculates SHAP values for each dataset, as used in the examples shown in the paper. Note that this script is only for demonstration purposes and will not be included in the future release of the CLE-SH library.

2. **analysis.py**

This script executes clesh.py and saves the output for each dataset.

At the beginning of the script, you can specify the list of datasets and adjust hyperparameters. Modify these settings to test with different datasets or parameters.

3. **comprehension.py**

This script generates the final report files based on the analysis results.

4. **clesh.py**

This script contains the core implementation of the CLE-SH algorithm.


## Execution Instructions
1.	Create a JSON configuration file:

    Place your configuration file in the /config directory. The test configuration is set to use the _test directory by default. This file includes multiple inputs, such as PATH, hyperparameters, and other settings. You can modify the PATH and other parameters in the configuration file to suit your setup.

2.	Prepare the dataset:

    Create a folder named data inside the directory specified by PATH. Place the following required files:
    - features.csv: The feature data for analysis.
    - shap.npy: The SHAP values to be used in the analysis.

3.	Write a shell script:

    Write a script to execute analysis.py and comprehension.py. You can refer to the example test shell scripts provided.

4.	Run the script:
    
    Execute the shell script to process the datasets and generate results.

## Test code for Reproduction execution guide 
### Metabolic Syndrome (MS)
[Dataset Link](https://www.kaggle.com/datasets/antimoni/metabolic-syndrome)
- Download CSV file and change it's name into data_original.csv
- execute script/test_MS.sh

### Heart Failure (HF)
[Dataset Link](https://doi.org/10.24432/C5Z89R)
- Download CSV file and change it's name into data_original.csv
- execute script/test_HF.sh

### Breast Cancer (BC)
[Dataset Link](https://doi.org/10.24432/C5GK50)
- Download feature and label npy file. Change it's name into features_original.npy, label_original.npy.
- execute script/test_default.sh

### Diabetic Retinopathy (DR)
[Dataset Link](https://doi.org/10.24432/C5XP4P)
- Download feature and label npy file. Change it's name into features_original.npy, label_original.npy.
- execute script/test_default.sh

### Inflammatory Bowel Disease (IBD)
[Dataset Link](https://doi.org/10.1093/gigascience/giad083)
- Download feature and label npy file. Change it's name into features_original.npy, label_original.npy.
- execute script/test_default.sh