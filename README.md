#DSE-230 Final Project Setup Guide

### 1. For running the code on local machine:

#### 1.1 Download and unzip datasets.zip into working directory (<work_dir>).

#### 1.2 Place CSV files from datasets.zip into HDFS: 
```
hadoop fs -copyFromLocal <work_dir>/BRFSS_2020_main_dataset.csv /;  
hadoop fs -copyFromLocal <work_dir>/Behavioral_Risk_Factor_Surveillance_System__BRFSS__Historical_Questions.csv /;  
hadoop fs -copyFromLocal <work_dir>/BRFSS_feature_codes_map.csv /; 
```

#### 1.3. Run the code in JupyterLab or other Python environment


### 2. Description of Files:  
#### 2.1. Datasets:  
BRFSS_2020_main_dataset.csv. ...  
Behavioral_Risk_Factor_Surveillance_System__BRFSS__Historical_Questions.csv ...  
BRFSS_feature_codes_map.csv. ...  

#### 2.2. Code: 
team3_project_notebook.ipynb ...   
team3_project_notebook.ipynb ...   

#### 2.3. Presentations:  
team3_project_proposal_presentation.pdf ...  
team3_project_final_presentation.pdf ...  


