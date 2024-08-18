import pathlib
from collections import Counter
import pickle

from pystempel import Stemmer

from src.basic_preprocessing import regex_preprocessing, remove_stop_words

stemmer = Stemmer.polimorf()


def main():
    raw_data_path = pathlib.Path('./data/raw')
    for file_path in raw_data_path.iterdir():
        raw_text = file_path.read_text()
        lowercase_text = raw_text.lower()
        text_regex_cleaned = regex_preprocessing(lowercase_text)
        preprocessed_path = pathlib.Path('data/preprocessed')
        preprocessed_path = preprocessed_path / file_path.name
        preprocessed_path.write_text(text_regex_cleaned)
        split_words = text_regex_cleaned.split()
        split_words = remove_stop_words(split_words)
        stemmed_words = [stemmer(word) for word in split_words]

        word_count = Counter(stemmed_words)

        counter_storing_path = './data/counters/' + file_path.name
        with open(counter_storing_path, 'wb') as storage_file:
            pickle.dump(word_count, storage_file)

    # Lemmatizing:


main()
