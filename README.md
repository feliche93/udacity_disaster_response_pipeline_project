# Disaster Response Pipeline Project

### Summary:
In this project I created a web app where an emergency worker can input a new message and get classification results in several categories. Furthermore, the web app will also display visualizations of the data.

![Screenshot of App](https://github.com/feliche93/udacity_disaster_response_pipeline_project/blob/master/Screenshot%202020-04-04%20at%2022.00.43.png)

### Project Components:

1. ETL Pipeline (process_data.py)
A Python script which is a data cleaning pipeline:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline (train_classifier.py)
A Python script which is a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App (run.py)
A web framework that serves the model and allows a user to get messages classified.

### Installation:
1. Clone the gitrepository
2. In the root directory create a virtual environment by running the following command in your terminal: `python3 -m venv env`
3. Activate environemnt by running the following in your terminal: `source env/bin/activate`
4. Install requirements in your terminal via `pip install -r requirements.txt`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    
    Please noe that currently running the ML pipeline takes quite some time so either take out more features or let it run for a longer time!

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
