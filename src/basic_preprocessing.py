import re

from src.stopwords import stopwords


def regex_preprocessing(text):
    text = re.sub('\(.*\)', ' ', text)
    text = re.sub(' +([a-z]+\.)+', '', text)
    text = re.sub(r"[^ a-ząęłżźćśó0-9\n]+", '', text)
    text = re.sub(r"\n", ' ', text)
    text = re.sub(r"[0-9]+", '', text)
    text = re.sub(" +", ' ', text)
    return text


def remove_stop_words(words_list):
    return [word for word in words_list if word not in stopwords]
