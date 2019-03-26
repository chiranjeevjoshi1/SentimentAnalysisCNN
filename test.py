import os
import re
import pickle
import tkinter
from tkinter import *

import numpy as np
from keras.models import model_from_json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from gensim.models.keyedvectors import KeyedVectors
import enchant

top = tkinter.Tk()
frame = Frame(top)
frame.pack()

sentence = StringVar()
result1 = StringVar()
result2 = StringVar()

sentence.set("Please Press Next to see results..")
result1.set('Positive')
result2.set('Negative')


# load json and create model
with open('saved_model/cnn_sentiment.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("saved_model/cnn_sentiment.h5")

model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=[
    'accuracy'])

test_file = open('test_sentences.txt', "r")

word_vectors = KeyedVectors.load_word2vec_format(
        '/home/john/geek_stuff/Data_Set/NLP/Google_News_corpus/GoogleNews-vectors-negative300.bin',
        binary=True, limit=None)


def get_wordnet_pos(treebank_tag):

    ## Removed Adjective from pos tagging as word_lemmatizer convert superlative degree to original form
    # if treebank_tag.startswith('J'):
    #     return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def pre_process(sentence):
    word_lemmatizer = WordNetLemmatizer()
    enchant_dict = enchant.Dict("en_US")

    clean_sentence = re.sub('[^A-Za-z ]+', '', sentence)

    word_max = 25
    word_count = 0
    words_to_keep = []
    for word, typ in nltk.pos_tag(word_tokenize(clean_sentence)):
        if word_count >= word_max:
            break
        typ = get_wordnet_pos(typ)
        if typ:
            lemmatized_word = word_lemmatizer.lemmatize(word, typ).lower()
        else:
            lemmatized_word = word_lemmatizer.lemmatize(word).lower()

        # Removing Stop words and correct spelled words
        # Remove all non-english or mis-spelled words
        with open('data/stop_words.pkl', 'rb') as file:
            stop_words = pickle.load(file)
        if enchant_dict.check(lemmatized_word):
            words_to_keep.append(lemmatized_word)
            word_count += 1

    # if number of words are less than max allowed word add padding
    if word_count < word_max:
        words_to_keep += ['<pad>'] * (word_max - word_count)

    word2vec_array = np.zeros([1, word_max, 300], np.float64)
    senti_vec_array = np.zeros([1, word_max, 3], np.float64)

    for word_idx, each_word in enumerate(words_to_keep):
        # Get sentiwordnet word vector
        senti_syns = list(sentiwordnet.senti_synsets(each_word))
        if senti_syns:
            over_all_score = np.zeros([1, 3])
            for val in senti_syns:
                over_all_score += np.array(
                    [val.pos_score(), val.neg_score(),
                     val.obj_score()])

            over_all_score /= len(senti_syns)
        else:
            over_all_score = np.array([0, 0, 1])

        senti_vec_array[0, word_idx] = over_all_score

        # Get googles word2vector
        try:
            word2vec_array[0, word_idx] = word_vectors[
                each_word]
        except Exception as exc:
            word2vec_array[0, word_idx] = np.random.uniform(
                low=-0.25, high=0.25, size=(1, 300))

    return [word2vec_array, senti_vec_array]


def sentiment_test():

    line = test_file.readline()
    if line:
        data = pre_process(sentence=line)
        score = model.predict(data)
        sentence.set(line)
        result1.set('Positive {0}%'.format(int(score[0][1]*100)))
        result2.set('Negative {0}%'.format(int(score[0][0]*100)))
    else:
        sentence.set('No more Texts Available. please close the window.')
        result1.set('Positive')
        result2.set('Negative')


# text = Text(top)
# text.insert(INSERT, "Hello.....")

message1 = Message(frame, textvariable=sentence, width='30c', bd=15)


message2 = Message(frame, textvariable=result1, width='10c',
                   bg='green')
message3 = Message(frame, textvariable=result2, width='10c',
                   bg='red')

B = tkinter.Button(top, text="Next", command=sentiment_test, justify='right')
B.pack()
message1.pack(side=TOP)
message2.pack(side=LEFT)
message3.pack(side=RIGHT)
top.mainloop()