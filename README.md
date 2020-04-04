# Disaster Response Pipeline Project

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
