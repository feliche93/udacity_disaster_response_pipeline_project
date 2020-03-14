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
from nltk.stem import LancasterStemmer, WordNetLemmatizer
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
import pickle

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


class PreProcessText(BaseEstimator, TransformerMixin):

    def remove_between_square_brackets(self, text):
        return re.sub(r'\[[^]]*\]', '', text)

    def replace_contractions(self, text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    def tokenize(self, text):
        text = self.remove_between_square_brackets(text)
        text = self.replace_contractions(text)
        tokens = word_tokenize(text)
        return tokens

    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

    def stem_words(self, words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def normalize(self, words):
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
        words = self.replace_numbers(words)
        words = self.remove_stopwords(words)
        return words

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).apply(self.tokenize)
        X = X.apply(self.normalize)
        X = X.apply(lambda x: ' '.join(x))

        return X


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
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )

    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        # 'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        # 'features__text_pipeline__tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [50, 100, 200],
        # 'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create grid search object
    cv = GridSearchCV(estimator=full_pipeline, param_grid=parameters)

    return cv


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
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


def main():
    # if len(sys.argv) == 3:
    if True:
        # database_filepath, model_filepath = sys.argv[1:]
        database_filepath, model_filepath = (
            'data/DisasterResponse.db', 'models/classifier.pkl')

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
