import re

from src.stopwords import stopwords

roman_numerals = re.compile(r'\s+m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})\s+')


def regex_preprocessing(text):
    text = re.sub(r'\(.*\)', ' ', text)
    text = re.sub(r'\s([a-z]\.)+', ' ', text)
    text = re.sub(r'[^ a-ząćęłńóśżź]', ' ', text)

    while roman_numerals.search(text):
        text = roman_numerals.sub(' ', text)

    text = re.sub(r'\s+', ' ', text)  # to don't have problem with ' ' in list returned by split
    return text


def remove_stop_words(words_list):
    return [word for word in words_list if word not in stopwords]
