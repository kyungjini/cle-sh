# cle-sh
This file is the code and data source file for the paper, "CLE-SH: Comprehensive Literal Explanation package for SHapley values by statistical validity". 

This code includes not only the library of CLE-SH, but also example testing illustrated in the paper. 

To execute the result as shown in the paper, execute analysis.py and comprehension.py to get final results, figures and the report.

-------------------------------
Here are the details of each file in the folder, _code. 

1. data_learning.py
: This file is to calculate SHAP values from each dataset, which are used to obtain examples illustrated in the paper. This is only to make examples for the paper, which means, will not be included in the future release of CLE-SH library. 

2. analysis.py
: This file is to execute clesh.py and save outputs from each dataset. 
: At the beginning, there is the list of datasets to analyze and hyperparameters. You can change it to get different results, or from another independent dataset. 

3. comprehension.py
: This file is to generate report file. 

4. clesh.py
: This file contains the source code of CLE-SH. 