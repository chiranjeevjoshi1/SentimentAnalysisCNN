import pickle
import time
import os

import argparse
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import sentiwordnet
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from load_data import load_movie_review_data
from pre_processing import pre_process, set_params
from sentiment_model import cnn_model


def get_data_by_batch(word_array, sentiment, batch_size, max_word, n_classes,
                      shuffle=True):
    # Load google's Pre-trained word to vector
    word_vectors = KeyedVectors.load_word2vec_format(
        '/home/john/geek_stuff/Data_Set/NLP/Google_News_corpus/GoogleNews-vectors-negative300.bin',
        binary=True, limit=None)
    len_data = len(word_array)
    start_index = 0

    while True:
        if start_index + batch_size > len_data:
            start_index = 0
            if shuffle:
                perm = np.arange(len_data)
                np.random.shuffle(perm)
                word_array = word_array[perm]
                sentiment = sentiment[perm]

        word2vec_array = np.zeros([batch_size, max_word, 300], np.float64)
        senti_vec_array = np.zeros([batch_size, max_word, 3], np.float64)
        req_target = sentiment[start_index: start_index + batch_size]
        for batch_idx, each_word_array in enumerate(
                word_array[start_index:start_index + batch_size]):
            # print('Word array ', each_word_array)
            # print('sentiment ', req_target[batch_idx])
            for word_idx, each_word in enumerate(each_word_array):
                # Get sentiwordnet word vector
                # print('each word ', each_word)
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
                # print('Ovear all score ', over_all_score)
                senti_vec_array[batch_idx, word_idx] = over_all_score
                # print(senti_vec_array)
                # Get googles word2vector
                try:
                    word2vec_array[batch_idx, word_idx] = word_vectors[
                        each_word]
                except Exception as exc:
                    word2vec_array[batch_idx, word_idx] = np.random.uniform(
                        low=-0.25, high=0.25, size=(1, 300))

        sess = tf.Session()
        target = sess.run(tf.one_hot(req_target, n_classes))
        start_index += batch_size
        yield ([word2vec_array, senti_vec_array], target)


# Params
max_word = 25
# Percent of data for train validation and test
train_pct, val_pct, test_pct = 0.9, 0.0, 0.10
batch_size = 32
n_class = 2
n_epoch = 10

# Other
saving_dir = 'saved_model'

# Load Movie Review Data
df = load_movie_review_data()
set_params(max_word=max_word)
word_array, sentiment = pre_process(df)

# Divide data into train, validation and test set
len_data = word_array.shape[0]
n_train_data = int(len_data * train_pct)
train_input = word_array[:n_train_data]
train_target = sentiment[:n_train_data]
val_data_index = int(len_data * (train_pct + val_pct))
n_val_data = val_data_index - n_train_data
val_input = word_array[n_train_data:val_data_index]
val_target = sentiment[n_train_data:val_data_index]
n_test_data = len_data - (n_train_data + n_val_data)
test_input = word_array[val_data_index:]
test_target = sentiment[val_data_index:]
# free memory space
word_array, sentiment = [], []

# load model
model = cnn_model(max_word)

# Load/Save data from checkpoint
callbacks_list = None

# if os.path.exists(saving_dir+"/cnn_sentiment.h5"):
#     # Loading Saved Weights
#     model.load_weights(saving_dir+"/cnn_sentiment.h5")
# else:
#     ## Saving Model at every epoch
#     checkpoint = ModelCheckpoint(saving_dir, verbose=1, save_best_only=False, save_weights_only=True, period=15)
#     callbacks_list = [checkpoint]

# Train Model
train_data_generator = get_data_by_batch(train_input, train_target, batch_size,
                                         max_word, n_class,
                                         shuffle=True)  # Get train data
model.fit_generator(train_data_generator,
                    steps_per_epoch=n_train_data // batch_size,
                    nb_epoch=n_epoch, callbacks=callbacks_list,
                    verbose=1, validation_data=None, validation_steps=None)

# Save train model
model_json = model.to_json()
with open(saving_dir + '/cnn_sentiment.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(saving_dir + "/cnn_sentiment.h5")
print("Saved model to {0}".format(saving_dir))

# test Model
batch_size = n_test_data
test_data_generator = get_data_by_batch(test_input, test_target, batch_size,
                                        max_word, n_class, shuffle=True)
score, acc = model.evaluate_generator(test_data_generator,
                                      steps=n_test_data // batch_size)
print('Score: {0}, Accuracy: {1}'.format(score, acc))
