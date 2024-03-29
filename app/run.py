from sklearn.externals import joblib
from plotly.graph_objs import Bar, Histogram
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from joblib import dump, load
from flask import Flask, jsonify, render_template, request
import plotly
import pandas as pd
from pathlib import Path
import sys
import json
from custom_pipelines import PreProcessText
from sqlalchemy import create_engine
from sqlalchemy import inspect

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

def load_data(database_filepath):
    base_dir = Path(__file__).resolve().parent.parent
    db = base_dir.joinpath(Path(database_filepath))
    engine = create_engine('sqlite:///' + db.as_posix())
    inspector = inspect(engine)

    print('Reading in data from {} Table...'.format(
        inspector.get_table_names()[0]))

    df = pd.read_sql("SELECT * FROM Messages", con=engine)

    for column in df.iloc[:, 5:].columns:
        df[column] = pd.to_numeric(df[column])

    return df


df = load_data(database_filepath='data/DisasterResponse.db')

# load model
model = load(
    "/Users/felixvemmer/Desktop/udacity_disaster_response_pipeline_project/models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    message_length = df['message'].apply(lambda x: len(x))
    category_counts = df.iloc[:, 5:].sum()
    category_names = list(df.iloc[:, 5:].columns)

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
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Category Counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=message_length,
                )
            ],

            'layout': {
                'title': 'Length of Messages',
                'xaxis': {
                    'title': "Message Length"
                }
            }
        },
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
