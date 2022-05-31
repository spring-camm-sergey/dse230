CONTENTS OF THIS FILE
---------------------

1. Project Title
2. Project Description
3. Converted Datasets, Code and Presentations GIT Repository
4. How to Install and Run the Project
5. Credits



1. Project Title:
------------------

DIABETES RISK PREDICTION FROM PERSONAL HEALTH INDICATORS



2. Project Description:
-----------------------

CDC's Behavioral Risk Factor Surveillance System (BRFSS) is the nation’s premier system of health-related
telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic
health conditions, and use of preventive services. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted
health survey system in the world. This project uses BRFSS 2020 dataset to predict the risk of diabetes disease in individual given his responses to the survey.

For more information on BRFSS phone survey, please visit following website:
- https://www.cdc.gov/brfss/index.html

For original Dataset Files, Scripts and Description, please visit following website: 
- https://www.cdc.gov/brfss/annual_data/annual_2020.html



3. Converted Datasets, Code and Presentations GIT Repository:
-------------------------------------------------------------

To download the code, datasets and project presentations please visit GitHub location:
- https://github.com/spring-camm-sergey/dse230

Files Description:

Datasets (datasets.zip) contains:
- BRFSS_2020_main_dataset.csv							this is the main dataset, downloaded from CDC website as SAS file (https://www.cdc.gov/brfss/annual_data/2020/files/LLCP2020XPT.zip) and converted into .csv in SAS Studio.
- Behavioral_Risk_Factor_Surveillance_System__BRFSS__Historical_Questions.csv	this file contains features codes, corresponding questions and possible answers.
- BRFSS_feature_codes_map.csv 							this file contains features codes and corresponding questions in the short form.


Code:
- dse230_project_notebook.ipynb							the notebook with the script and (almost) all cells executed
- dse230_project_notebook.py							the Python script generated from the notebook
- dse230_project_notebook.pdf							exported PDF of the notebook


Presentations:
- project_proposal_presentation.pdf						project proposal presentation PDF
- project_final_presentation.pdf						final project presentation PDF



4. How to Install and Run the Project:
--------------------------------------
- All files are located in project GIT repository: https://github.com/spring-camm-sergey/dse230

a. Download and unzip datasets.zip into your working directory (<work_dir>).

b. Place all CSV files from the previous step into HDFS. Example:
	hadoop fs -copyFromLocal <work_dir>/BRFSS_2020_main_dataset.csv /;  
	hadoop fs -copyFromLocal <work_dir>/Behavioral_Risk_Factor_Surveillance_System__BRFSS__Historical_Questions.csv /;  
	hadoop fs -copyFromLocal <work_dir>/BRFSS_feature_codes_map.csv /; 


c. Download the code file (.py or .ipynb)

d. Run the code:
- Jupiter Notebook (Lab): open the dse230_project_notebook.ipynb file and click Run > Run All Cells
- Python console / IDE:  run dse230_project_notebook.py script

e. Clean-up HDFS:

	hadoop fs -rm /BRFSS_2020_main_dataset.csv;
	hadoop fs -rm /BRFSS_feature_codes_map.csv;
	hadoop fs -rm /Behavioral_Risk_Factor_Surveillance_System__BRFSS__Historical_Questions.csv;


5. Credits:
-----------

The project initiated as part of UCSD's DSE230 final project coursework. 
The ML script was developed by below student maintainers and under the guidance of professor M. H. Nguyen (PhD). 

Current Maintainers: 									Sergey Gurvich <sgurvich@ucsd.edu>
											Chunxia Tong <chtong@ucsd.edu>
											Camm Perera <cperera@ucsd.edu>  
