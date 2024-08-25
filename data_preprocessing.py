import pathlib
from collections import Counter
import pickle

from pystempel import Stemmer

from src.basic_preprocessing import (regex_preprocessing, remove_stopwords_before_stemming,
                                     remove_stopwords_after_stemming)

stemmer = Stemmer.polimorf()


def main():
    raw_data_path = pathlib.Path('./data/raw')

    for file_path in raw_data_path.iterdir():
        text = file_path.read_text()
        text = text.lower()

        text = regex_preprocessing(text)

        preprocessed_path = pathlib.Path('data/preprocessed')
        preprocessed_path = preprocessed_path / file_path.name
        preprocessed_path.write_text(text)

        split_words = text.split()
        split_words = remove_stopwords_before_stemming(split_words)
        stemmed_words = [stemmer(word) for word in split_words]
        stemmed_words = remove_stopwords_after_stemming(stemmed_words)

        word_count = Counter(stemmed_words)

        counter_storing_path = './data/counters/' + file_path.name
        with open(counter_storing_path, 'wb') as storage_file:
            pickle.dump(word_count, storage_file)


main()
