import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle

def load_src(name, filepath):
    """
    Load a python file from another folder
    --
    Inputs:
        name: name of the imported file
        filepath: python file to be imported
    Outputs:
        df: the combined dataframe
    """
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), filepath))

load_src("classifier", "../models/train_classifier.py")
from classifier import TextSelector,ColumnExtractor,DummyTransformer,tokenize




app = Flask(__name__)


# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
#model = joblib.load(open('./models/filename.pkl','rb'))
model = pickle.load( open( "./models/classifier.p", "rb" ) )

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    related_count = df.related.value_counts()
    related_names = list(df.related.unique())
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_count = list(df[df.columns[4:]].sum())
    category_name = df.columns[4:]   
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
         {
            'data': [
                Bar(
                    x=related_names,
                    y=related_count,
                     marker=dict(
                        color=['rgba(222,45,38,0.8)','rgba(107, 107, 107,1)'])
                )
            ],

            'layout': {
                'title': 'Disaster Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related"
                }
                
            }
            
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                     marker=dict(
                        color=['rgba(107, 107, 107,1)', 'rgba(222,45,38,0.8)',
                            'rgba(204,204,204,1)'])
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
                    x=category_name,
                    y=category_count,
                     marker=dict(
                        color=['rgba(222,45,38,0.8)','rgba(107, 107, 107,1)', 
                            'rgba(204,204,204,1)'])
                )
            ],

            'layout': {
                'title': 'Distribution of Target Features',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Features",
                    'showticklabels':True,
                    'tickangle':315,
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
    message = request.args.get('message', '') 
    genre = request.args.get('genre', '') 
    query = pd.DataFrame(columns=['message','genre'])
    query.loc[0] = [message,genre]
    
    # use model to predict classification for query
    classification_labels = model.predict(query)[0]
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