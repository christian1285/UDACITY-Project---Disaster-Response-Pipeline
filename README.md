# UDACITY-Project---Disaster-Response-Pipeline


## Introduction

In this project we will classify real messages that were sent during disaster events by various users. You will be We create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project included a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## Project Components
There are three components included.

1. ETL Pipeline
Python script process_data.py contains a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
Python script train_classifier.py contains a a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
Python script run.py contains a web app that:

- Loads data from the trained model
- Allows to classify new messages with the trained model
- contains several visualisations from the training data 

## Instructions
The python script can be executed within the terminal with the following parameters:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
run.py

## Further remarks

Within the development process several different classification algorithms were tested with various parameter settings (RandomForestClassifier, MLPRegressor, LinearSVC). Further characteristics like word count character count were added but without any increase of model performance. It turned out that LinearSVC by far provides the best F1 scores out of all models. Only one hyperparameter is used in the grid search here due to performance resasons and none of them lead to any improvement of the model performance.

The model provides weak performance for some of the categories with very few training data - here more training data is required. In order to directly find the categories which need to be improved the last visualisation was added.

The visualisation shows the share of category tags in the training data set and the corresponsing precision score. High category shares and low precision score could show poor model performance which ould be enhanced by further characteristics etc. whereas  for low category shares more training data should improve the results.

