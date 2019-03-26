from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Merge, Dropout
from keras import regularizers


def cnn_model(max_len):
    """
    :param word2vec: Googles word to vector o/p for a given batch
    :param senti_word_net_vec: word vector obtained from senti-word-net for the batch
    :param target: target sentiment
    :return:
    """
    # Word2vec CNN model
    model1 = Sequential()
    model1.add(Conv1D(100, 5, activation='relu',
                      kernel_regularizer=regularizers.l2(0.003),
                      padding='same', input_shape=(max_len, 300)))
    model1.add(Dropout(0.5))
    model1.add(Conv1D(100, 4, activation='relu',
                      kernel_regularizer=regularizers.l2(0.003),
                      padding='same'))
    model1.add(Dropout(0.5))
    model1.add(Conv1D(100, 3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.003),
                      padding='same'))
    model1.add(Dropout(0.5))
    model1.add(MaxPool1D(2, strides=2))

    # Sentiword net CNN model
    model2 = Sequential()
    model2.add(Conv1D(100, 5, activation='relu',
                      kernel_regularizer=regularizers.l2(0.003),
                      padding='same', input_shape=(max_len, 3)))
    model2.add(Dropout(0.5))
    model2.add(Conv1D(100, 4, activation='relu',
                      kernel_regularizer=regularizers.l2(0.003),
                      padding='same'))
    model2.add(Dropout(0.5))
    model2.add(Conv1D(100, 3, activation='relu',
                      kernel_regularizer=regularizers.l2(0.003),
                      padding='same'))
    model2.add(Dropout(0.5))
    model2.add(MaxPool1D(2, strides=2))

    # Merge both layer
    merge_model = Merge([model1, model2], mode='concat')

    # Final model with softmax classifier
    model_final = Sequential()
    model_final.add(merge_model)
    model_final.add(Flatten())
    model_final.add(Dropout(0.5))
    model_final.add(Dense(200, activation='relu',
                          kernel_regularizer=regularizers.l2(0.003)))
    model_final.add(Dropout(0.5))
    model_final.add(Dense(2, activation='softmax'))
    model_final.compile(loss='binary_crossentropy', optimizer='adagrad',
                        metrics=['accuracy'])

    return model_final


if __name__ == '__main__':
    cnn_model(20)
