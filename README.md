# DSE-230 Final Project Setup Guide

### 1. For running the code on local machine:

#### 1.1 Download and unzip datasets.zip into your working directory (<work_dir>).

#### 1.2 Place all CSV files from the previous step into HDFS: 
```
hadoop fs -copyFromLocal <work_dir>/BRFSS_2020_main_dataset.csv /;  
hadoop fs -copyFromLocal <work_dir>/Behavioral_Risk_Factor_Surveillance_System__BRFSS__Historical_Questions.csv /;  
hadoop fs -copyFromLocal <work_dir>/BRFSS_feature_codes_map.csv /; 
```

#### 1.3. Run the code in JupyterLab or another Python environment


### 2. Description of Files:  
- README.md: this file
- README.txt: txt version of this guidance

#### 2.1. Datasets:  
- BRFSS_2020_main_dataset.csv: this is the main dataset, downloaded from CDC website as SAS file (https://www.cdc.gov/brfss/annual_data/2020/files/LLCP2020XPT.zip) and converted into .csv in SAS Studio.    

- Behavioral_Risk_Factor_Surveillance_System__BRFSS__Historical_Questions.csv:  this file contains features codes, corresponding questions and possible answes.

- BRFSS_feature_codes_map.csv: this file contains features codes and corressponding questions in the short form.

#### 2.2. Code: 
- dse230_project_notebook.ipynb: the notebook with the script and (almost) all cells executed   
- dse230_project_notebook.py: the Python script generated from the notebook  
- dse230_project_notebook.pdf: exported PDF of the notebook

#### 2.3. Presentations:  
- team3_project_proposal_presentation.pdf: project proposal presentation PDF
- team3_project_final_presentation.pdf: final project presentation PDF


Team: Sergey Gurvich, Camm Perera, Chunxia Tong

