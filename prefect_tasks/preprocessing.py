"""Function for basic preprocessing - remove non letter chars, and make stemming, also make words counts."""
from pathlib import Path
import pickle
import gzip
import json
from collections import Counter

import xattr
from prefect import task

from src.basic_preprocessing import (regex_preprocessing, remove_stopwords_before_stemming,
                                     remove_stopwords_after_stemming, stem_words)


@task
def make_words_counts(court_type: str, appeal: str, court_name: str):
    preprocessing_base_path = Path('data') / 'preprocessed'
    counters_path =  Path('data') / 'counters'

    preprocessing_particular_dir = preprocessing_base_path / court_type / appeal / court_name

    for file_path in preprocessing_particular_dir.iterdir():
        with gzip.open(file_path, 'rb') as preprocessing_file:
            text = preprocessing_file.read()

        text = text.decode()

        split_words = text.split()
        split_words = remove_stopwords_before_stemming(split_words)
        stemmed_words = stem_words(split_words)
        stemmed_words = remove_stopwords_after_stemming(stemmed_words)

        word_count = Counter(stemmed_words)

        counter_storing_path = counters_path / f'{file_path.stem}.pickle'
        attributes = {'court_type': court_type, 'appeal': appeal, 'court_name': court_name}
        attributes = json.dumps(attributes)
        with open(counter_storing_path, 'wb') as storage_file:
            pickle.dump(word_count, storage_file, protocol=pickle.HIGHEST_PROTOCOL)
            xattr.setxattr(storage_file, 'user.attributes', attributes.encode())


@task
def preprocessed_justification(court_type: str, appeal: str, court_name: str):
    justification_base_path = Path('data') / 'justification'
    preprocessing_base_path = Path('data') / 'preprocessed'

    justification_particular_dir = justification_base_path / court_type / appeal / court_name
    preprocessing_particular_dir = preprocessing_base_path / court_type / appeal / court_name

    for file_path in justification_particular_dir.iterdir():
        with gzip.open(file_path, 'rb') as justification_file:
            text = justification_file.read()
        text = text.decode()
        text = text.lower()

        text = regex_preprocessing(text)

        preprocessed_path = preprocessing_particular_dir / file_path.name

        try:
            with gzip.open(preprocessed_path, 'wb') as preprocessed_file:
                preprocessed_file.write(text.encode())
        except FileNotFoundError:
            preprocessing_particular_dir.mkdir(parents=True)
            with gzip.open(preprocessed_path, 'wb') as preprocessed_file:
                preprocessed_file.write(text.encode())
