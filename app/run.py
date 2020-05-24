import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)
df_vis = pd.read_sql_table('VisualisationTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ''' 2. Visualisation -  Category tags per column in training dataset ''' 
    
    Y = df[df.columns[4:]]
    #get number of tags (value 1) per column
    tags_per_column = []
    for col in Y.columns:
        tags_per_column.append(Y[col].sum())

    #create sorted dataframe
    df_cat_sorted = pd.DataFrame({'count':tags_per_column, 'category_name':Y.columns}).sort_values('count', ascending=False)
    count_categories = df_cat_sorted['count']
    category_names = df_cat_sorted['category_name']

    ''' 3. Visualisation -  Category tags per column in training dataset ''' 
    category_names = df_vis['category name']
    y1 = df_vis['precision score']
    y2 = df_vis['share in train']
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }  
            
        },
                {
            'data': [
                Bar(
                    x=category_names,
                    y=count_categories
                )
            ],

            'layout': {
                'title': 'Category tags per column in training dataset ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }  
            
        },
               {
            'data': [
               Bar(
                   x=category_names,
                   y=y1,
                   name='Points',            
                   marker=dict(color= 'rgba(220,49,72, 0.8)', line= dict(width= 1)),
                   yaxis='y'
                ),
                Scatter(
                    x=category_names, 
                    y=y2, 
                    marker= dict(line= dict(width= 1), 
                           size= 8), 
                    line=dict(color= '#B0122C', width= 1.5),
                    name= 'Accumulated pts',
                    yaxis='y2'
                    )
            ],

            'layout': {
                'title': 'test ',
                'yaxis': {
                    'title': "y axis 1"
                },
                'yaxis2': {
                    'title': "y axis 2",
                    'overlaying': "y",
                    'side': "right"
                },
                'xaxis': {
                    'title': "x axis"
                }
            }  
            
        }
        
        
        
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()