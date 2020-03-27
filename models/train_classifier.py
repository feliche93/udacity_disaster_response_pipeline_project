import os
import re
import string
import sys
import unicodedata

import contractions
import inflect
import nltk
import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine
from sqlalchemy import inspect
from textblob import TextBlob
from pathlib import Path
from joblib import dump, load
from custom_pipelines import PreProcessText

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    base_dir = Path(__file__).resolve().parent.parent
    db = base_dir.joinpath(Path(database_filepath))
    engine = create_engine('sqlite:///' + db.as_posix())
    inspector = inspect(engine)

    print('Reading in data from {} Table...'.format(
        inspector.get_table_names()[0]))

    df = pd.read_sql("SELECT * FROM Messages", con=engine)

    X = df['message'].values
    y = df.iloc[:, 4:].values
    y = y.astype(int)
    category_names = df.iloc[:, 4:].columns

    return X, y, category_names


def tokenize(text):
    pass


def build_model():
    text_pipeline = Pipeline(
        steps=[
            ('text_pre_process', PreProcessText()),
            ('vect', CountVectorizer()),
            ('transform', TfidfTransformer())
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            ('text_pipeline', text_pipeline),
            ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=True)))
        ]
    )

    parameters = {
        # 'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        # 'text_pipeline__vect__max_features': (None, 5000, 10000),
        # 'text_pipeline__transform__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [50, 100, 200],
        # 'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create grid search object
    cv = GridSearchCV(estimator=full_pipeline, param_grid=parameters)

    return cv

    # return full_pipeline


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)

    for y_pred_category, y_test_category, category_name in zip(y_pred.T, y_test.T, category_names):
        print(category_name)
        print(
            classification_report(
                y_true=y_test_category,
                y_pred=y_pred_category
            )
        )


def save_model(model, model_filepath):
    base_dir = Path(__file__).resolve().parent.parent
    model_filepath = base_dir.joinpath(Path(model_filepath))
    dump(model, model_filepath.as_posix())


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
