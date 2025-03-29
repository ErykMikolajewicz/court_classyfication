import re
import os

import src.stopwords as stopwords


roman_numerals = re.compile(r'\s+m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})\s+')


def regex_preprocessing(text):
    text = re.sub(r'\(.*\)', ' ', text)
    text = re.sub(r'\s([a-z]\.)+', ' ', text)
    text = re.sub(r'[^ a-ząćęłńóśżź]', ' ', text)

    while roman_numerals.search(text):
        text = roman_numerals.sub(' ', text)

    text = re.sub(r'\s+', ' ', text)  # to don't have problem with ' ' in list returned by split
    return text


def remove_stopwords_before_stemming(words_list):
    return [word for word in words_list if word not in stopwords.before_stemming]


def remove_stopwords_after_stemming(words_list):
    return [word for word in words_list if word not in stopwords.after_stemming]


STEMMER_TYPE = os.environ['STEMMER_TYPE']
match STEMMER_TYPE:
    case 'STEMPEL':
        from pystempel import Stemmer
        stemmer = Stemmer.polimorf()
    case 'MORFEUSZ':
        import morfeusz2
        morfeusz = morfeusz2.Morfeusz()
    case 'SPACY':
        import spacy
        spacy_nlp = spacy.load('pl_core_news_md')
    case _:
        raise ValueError('Invalid stemmer type!')


def stem_words(words: list[str]) -> list[str]:
    match STEMMER_TYPE:
        case 'STEMPEL':
            stemmed_words = [stemmer(word) for word in words]
            return stemmed_words
        case 'MORFEUSZ':
            stemmed_words = []
            for word in words:
                stemmed_word: str = morfeusz.analyse(word)[0][2][1]
                try:
                    index = stemmed_word.index(':')
                    stemmed_word = stemmed_word[:index]
                except ValueError:
                    pass
                stemmed_words.append(stemmed_word)
            return stemmed_words
        case 'SPACY':
            words = spacy_nlp(' '.join(words))
            stemmed_words = [word.lemma_ for word in words]
            return stemmed_words
        case _:
            raise ValueError('Invalid stemmer type!')
