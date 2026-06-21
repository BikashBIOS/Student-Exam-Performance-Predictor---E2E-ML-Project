# END 2 END MACHINE LEARNING PROJECT

## CREATE GITHUB REPO:
1. Create a git hub repo
2. Create your project folder - "ML"
3. open vs code in that folder
4. go to cmd terminal
5. create env -> python -m venv venv
6. inside venv -> git init , git add . , git commit, git push
7. You can see the readme file in your repository. 

## ADD YOUR REQUIREMENTS : 
1. Create a requirements.txt file. 
2. Mention all the required Python libraries that you required for building your project.
3. Create new setup.py file in ML. (setup.py - python script included with Python written libraries to ensure that the program is installed correctly). 
4. Configure all the code and get the requirements() by pulling all the required libraries by requirements.txt
5. Then create a 'src' folder in ML. Then create a new file - '__init__.py'
6. Push your code to GitHub.

## ADD FILES : 
1. In src, add components folder and then create : 
data_ingestion.py
data_transformation.py
model_trainer.py
2. In src, add pipeline folder and then create :
__init__py
exception.py
logger.py
utils.py

## ADD EXCEPTIONS:
1. In exceptions.py, write the necessary code to create custom exceptions - to return the exceptions if any , while writing the code. 
2. error_message_detail - error function to get the error from sys along with line number
3. Define the Custom exception class to retrieve the error from above function.

## ADD LOGGER:
1. In logger.py, write the necessary code for logging your important processes while going through out the project. 
2. In such complex projects involving 1000 processes, you need to track each and every process if they are completing fine or not. 
3. LOG_FILE - creates a time data log file in your logs folder. 
4. Code for creating the log path to create the LOG_FILE in your logs folder. 
5. logging.basicconfig - sets the rules for your logs on how it's written - enhances the time and date for each logging statement. 

## JUPYTER NOTEBOOK (EDA & MODEL TRAINING) 
1. Create notebook folder and data folder inside it. 
2. Put your dataset inside data folder. 
3. Create 2 .ipynb files in notebook folder. 1 for EDA and 1 for Model training. 
4. Write the necessary code for both the files : 
### EDA :
- Read the Data
- Data Cleansing
- Visualisation
### MODEL TRAINING : 
- Train test split
- Encode categorical values & Scale your numerical values
- define evaluate_model() for calculating mse, mae and r2
- Apply all ML models - predict data, get the above scores from evaluate_model()
- Compare the R2 scores for all the models
- Train your model on the best model and predict the data. 
- Plot/Compare Actual and Predicted graph and draw the best fit line. 


## DATA INGESTION : 
1. Above ipynb files were just for our learning purposes, but we need to apply that same logic in our data_ingestion.py file with professional project coding. 
2. It is the entry point of a typical Machine Learning pipeline. In industry terms, "Data Ingestion" is the process of taking raw data from a source and preparing it for the next stages.
3. dataclass is a concise way to create classes that primarily store data - acts as a folder manager - tells the code to save output into Artifacts folder. 
4. DataIngestion Class :
- Pass the dataclass into the ingestion_config.
- In initiate_data_ingestion() - read the data,
creates the artifacts folder (checks if exists or not), 
creates the csv file for raw data. 
- Train test split 
- Create the csv file for train data and test data.
- Return the train and test data path. 
5. Run the class using __main__ and initiate_data_ingestion() 



## DATA TRANSFORMATION
1. dataclass to save your preprocessor pickle file with all the rules to apply through out your code.
2. get_transformer_object() - create the pipeline to apply OneHotEncoder to categorical values and StandardScaler to the numerical values. Output is the preprocessor obj. 
3. initiate_transformation() - Take X and y - divide into training and test data. And then fit/transform with help of preprocessor. 
4. Here, after fit/transforming the training data and test data, we concatenate it with the target data(math score) using np.c_ (column wise concatenation). Then we get train_arr and test_arr.
5. Now your normalized data is now ready to use - train_arr, test_arr.
6. Save your preprocessor object in the pickle file. Use save_object() function that we need to write it in utils.py to call from there. Pass the file path from preprocessor_obj_path that we initialized in @dataclass to save in artifacts folder. (saving this for using it in web app for real time predictions). 
7. All things set, now to run this classes and functions, call it in the __main__ function in data_ingestion.py. Call the DataTransformation() and the initiate_data_transformation() with the train and test data we prepared in data_ingestion file. 


## MODEL TRAINER :
1. Create dataclass to initialize and save your pkl file in the artifacts folder when created.
2. Create ModelTrainer clas with initiate_model_trainer() function to input the train and test arr. 
3. Slice the X train and X test, y train and y test. Create the dict of models to apply. 
4. Train all your models based on your training and testing data in model_report with help of evaluate_models() which we have to add in utils.py - This evaluate_model() helps to train your data on that model, predict the values, and also evaluate the r2 score. 
5. This report_model then returns a dictionary where the keys are model names and the values are their R² scores.
6. Then find the best model score by evaluating the max score from model_report values.
7. Find the best model name based upon the keys for the best model score. 
8. save_object() - save your trained model config in the mentioned path in @dataclass.
9. predict your values in the best model and return the r2 score.
10. Now edit the main Data ingestion file. Import model_trainer.py and then call ModelTrainer() along with initiate_model_trainer(). 


## HYPER PARAMETER TUNING
1. Add the hyperparameters and implement GridSearchCV in utils.py (Comment the previous model training code as we are implementing the grid search training).
2. Pass the params in the evaluate_model().
3. Add the same params value in the model_report in model_trainer.py where you are calling evaluate_model().
4. Add the Hyperparameters of different models in a new dict - params in model_trainer.py.
5. Execute the data ingestion.py again with the new hyperparameter codes and it will give you a better output of your r2 score from the previous models. 


## FLASK (PREDICT PIPELINE)
1. Create app.py for our flask application in ml folder.
2. Initialize your / route to the index.html. (Create index.html in a templates folder in ml folder)(Add all the html files in the templates folder)
3. Create home.html in a templates folder, paste the required html code. It's a simple code to take input from the user for the independent data.
4. Call the predict_datapoint() initialized in app.py (/predictdata) in the home.html's form tag (in the action) with method as POST (to post the data to the model from the user).
5. Now create CustomData class in predict_pipeline.py in which you basically have to map all the input fields from html to our independent data.
6. Provide all your fields and map it to self. 
7. Then create predict() in PredictPipeline Class and pass the model.pkl and preprocessor.pkl path file and apply load_object with both the paths. (Initialize load_object() in utils.py - it basically helps to load the pickle files).
8. After loading, scale the data using preprocessor and then predict the data using the model. Return the preds.
9. Then you have to call this CustomData() in app.py in your (/predictdata) route in POST method to get the data from html/user.
10. Based on this data, call the get_data_as_data_frame() from CustomData() as pred_df
11. Then based on this pred_df, we can predict the data by calling PredictPipeline() based on our preprocessor and model.pkl files. 
12. Then post the results in the home.html - in h2 tag to get the results.
13. Then call the main function in 0.0.0.0 host in app.py.


## AWS Beanstalk Deployment
1. Create new folder .ebextensions. Add new file python.config and add necessary codes.
2. Change your app.py file name to application.py
3. Then create an Elastic Beanstalk app in AWS Console
4. Create Codepipeline in AWS Console. Provide your GitHub link and connect to your Code base.


## DOCKER Deployment/AWS CI-CD Pipelines
1. Create the image for your project on Docker desktop : 
- docker build student-performance-app
- docker run -p 5000:5000 student-performance-app
- if app runs successfully in localhost:5000, your Dockerfile is perfect.
- IMP - Add a new file 'Dockerfile' in your project folder and copy paste the generic code. 
2. Create a new folder '.github'. Inside it create 'workflows'. Then create main.yaml file.
3. Copy paste the code in main.yaml (The code present in your github repo -> Actions -> Deploy to Amazon ECS)
4. Create your IAM user in AWS IAM Users:
- Grant the permissions policy - AmazonEC2ContainerRegistryFullAccess and AmazonEC2FullAccess
- Create Access Key -> For CLI -> Create Access key -> Download and keep it.(User Created)
5. Go to ECR (elastic container registy):
- create new repository
- give name student-performance
- Create repository
6. Go to EC2 Instance:
- Give webserver name as 'student-performance'
- Select ubuntu 64
- Instance - t2.micro (free)
- Select Allow HTTP and Allow HTTPS
- Launch Instance
- After successfully running, connect to instance.
- Connect to EC2 Instance Connect.
- Now you will get the command line interface to give commands.
7. Install Docker packages in EC2 : 
- In CLI, sudo apt-get update -y
- sudo apt-get upgrade
- curl -fsSL https:get.docker.com -o get-docker.sh
- sudo sh get-docker.sh
- sudo usermod -aG docker ubuntu
- newgrp docker
- To check if everything's fine, run 'docker'
8. Configure 'Runner' 
- Open Github repo, go to Settings -> Actions -> Runner.
- Create Runner -> Linux -> Copy the code. 
- Paste in EC2 Instance CLI. 
- Execute all the commands of the Runner that's shown in github.
- Give the app runner name as 'self-hosted'
- Now ./run.sh -> Connected to Github. 
- After this , your self-hosted runner would be created in github runners and it would be running in green-idle.
9. Add the Secret Keys in your Github repo :
- Add a new key - AWS_ACCESS_KEY_ID. Open your downloaded access key in step 4, and copy the Access key in the Secret. Then Create.
- Add a new key - AWS_SECRET_ACCESS_KEY - Same copy paste the secret access key and create.
- Add 3rd key - AWS_REGION - In Secret - add your region for eg. us-east-1 and create.
- Add 4th key - AWS_ECR_LOGIN_URI - Copy your ECR instance url/your ecr app name that created in step 5.
- Add 5th key - ECR_REPOSITORY_NAME - the name you have provided to your ECR instance.
- Commit all the changes. 
10. Now if you make any changes in your code in your project, it will be registered in the workflows tab of your github.And it will directly start to deploy in AWS ECR. 
11. Go to our EC2 instance : 
- Open Security tab
- Edit Inbound rules
- Add 'Custom TCP' in port - 8080 (as mentioned in app.py in our project)
- Then open the url of your EC2 instance and add :8080 in that and run in browser - our project will successfully be displayed. 
- IMP - ensure to close all your EC2 instances, ECR repositories, Delete users and remove the Runners to avoid extra charges.


## DOCKER Deployment/AZURE
1. Go to Microsoft Azure -> Container Registry:
- Create Container Registry
- Give Resource group, Registry name, Create
- After creating, go to Access Keys, and enable the Admin user. Copy the Login server and Password.
2. Open Web App:
- Create Web App
- Give Resource group, Name, Select Docker Container, OS-Linux, Next.
- Options-Single Container, Select the same Registry and then we have to upload the Image.
3. Creating Docker Image for our project:
- Go to our project cmd and write : 
- docker build -t 'ContainerRegistry Login server/application name:latest'
- docker login 'ContainerRegistry Login server' (Provide your user name and ContainerRegistry password)
- docker push 'ContainerRegistry Login server/application name:latest'
- Now the full docker image file will be pushed to your Container Registry.
4. After you build the image, we need to upload it in the web app in step 2 and create web app. 
5. After deployment to the web app, Go to Deployment Center in your created web app. 
- enable Continuous deployment.
- Select the Source as 'Github Actions'
- Select organization as your Github user name
- Select the Repository name.
- Select the branch as mian. 
- Then save.
6. Then in our github, ./github/workflows will get created and now you can see your deployment/continuous changes will start showing in your workflows tab in github repo.
