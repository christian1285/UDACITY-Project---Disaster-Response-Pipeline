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
