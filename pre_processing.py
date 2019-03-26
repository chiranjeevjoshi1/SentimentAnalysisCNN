import html.parser
import re
import pickle

import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import enchant


word_max = 25


def set_params(max_word):
    global word_max
    word_max = max_word


def get_wordnet_pos(treebank_tag):
    ## Doesn't include Adjective in pos tagging as it is lemmatized to original form
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def pre_process(df):
    num_data = len(df)
    word_array = np.array([])
    sentiment = []
    for idx in range(num_data):
        # Escape HTML char if present
        html_parser = html.parser.HTMLParser()
        html_cleaned_data = html_parser.unescape(df['text'][idx])

        # Remove all unnecessary special character
        html_cleaned_data = re.sub('[^A-Za-z ]+', '', html_cleaned_data)

        # Performing Word Lemmatization on text
        word_lemmatizer = WordNetLemmatizer()
        words_to_keep = []
        word_count = 0
        for word, typ in nltk.pos_tag(word_tokenize(html_cleaned_data)):
            if word_count >= word_max:
                break

            typ = get_wordnet_pos(typ)
            if typ:
                lemmatized_word = word_lemmatizer.lemmatize(word, typ).lower()
            else:
                lemmatized_word = word_lemmatizer.lemmatize(word).lower()

            with open('data/stop_words.pkl', 'rb') as file:
                stop_words = pickle.load(file)
            # Removing Stop words and correct spelled words
            # Remove all non-english or mis-spelled words
            enchant_dict = enchant.Dict("en_US")
            if enchant_dict.check(lemmatized_word):
                words_to_keep.append(lemmatized_word)
                word_count += 1

        # if number of words are less than max allowed word add padding
        if word_count < word_max:
            words_to_keep += ['<pad>']*(word_max - word_count)

        if len(word_array) != 0:
            word_array = np.vstack((word_array, words_to_keep))
            sentiment.append(df['sentiment'][idx])
        else:
            word_array = np.hstack((word_array, words_to_keep))
            sentiment.append(df['sentiment'][idx])

    return word_array, np.array(sentiment)