import sys
from sqlalchemy import create_engine
import pickle
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.externals import joblib 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponseTable', con=engine)
    
    # set features and label X, Y and category names
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # setup pipeline with TFIDF and MultiOutput LinearSVC()
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(LinearSVC()))
        ])

    parameters = { 
         'clf__estimator__dual': [True, False]
    }    
    
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, scoring='f1_micro', cv=2, n_jobs=-1)
    
    return model

precision_score_categories = []
def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    # print classification report for each category
    
    for cat in category_names:
        print('---------')
        print('Category: '+cat)
        print(classification_report(Y_test[cat], Y_pred[cat]))
        print('---------')
        # save precision scores for each column (only one tags)
        precision_score_categories.append((float(classification_report(Y_test[cat], Y_pred[cat]).split()[10])))

        
def save_model(model, model_filepath):
    # Save the model as a pickle in a file 
    joblib.dump(model, model_filepath) 

def save_visualisation_data(df, database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('VisualisationTable', engine, if_exists='replace', index=False) 

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Visualisation data saved!')
        share_ones_train = []
        for col in Y_train:
            share_ones_train.append(Y_train[col].sum()/len(Y_train))
        df_vis = pd.DataFrame({'precision score':precision_score_categories, 'category name':category_names, 'share in train':share_ones_train})
        save_visualisation_data(df_vis, database_filepath)
        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()