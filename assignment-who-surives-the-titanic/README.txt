----------------------------------------
---------INTRODUCTION-------------------
----------------------------------------
Welcome to Survival Pedictor

This application receives a (titanic) dataset in a csv format with certain keys/columns, after which it will predict class Survived/Not Survived. 

The project folder assignment-who-surives-the-titanic contains the following structure:

########Folders##############
1. #Data#: this folder contains the dataset that is being used to train the model
2. #Prep-JupyterNotebook#: this folder contains an ipynb file in which the steps of data understanding, preprocessing and modeling are documenten (if you like to run the ipynb file then you are required to install the modules in jupyter-reqs.txt that is located in the same folder)
3. #static#: this folder contains all the static elements the app needs for the html pages. 
4. #templetes#: this folder contains html pages
5. #userfiles#: this folder is empty but once the user uploads the file it will uploaded into this folder. The results of prediction will be set into a csv and will be uploaded into this folder. The user can then download the results

#########Files###############

6. app.py: API for prediction model based on Flask
7. requirements.txt: this file contains all the modules that are needed by app.py. IT MUST BE INSTALLED. 


----------------------------------------
---------INSTRUCTIONS-------------------
----------------------------------------

WINDOWS
1. If you do not have Python installed in you computer, please first install python
2. Navigate to the project folder (assignment-who-surives-the-titanic), and run the following command: pip install -r requirements.txt
3. After the requirements are installed, run app.py with the following command: python app.py

#################################################################################################################################
 
