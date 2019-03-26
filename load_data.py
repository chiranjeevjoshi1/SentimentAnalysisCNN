
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize


def load_movie_review_data():
    file_path = 'rt-polaritydata/'
    pos_file = 'rt-polarity.pos'
    neg_file = 'rt-polarity.neg'

    df = pd.DataFrame()
    n_word_count = {}
    for file in [pos_file, neg_file]:
        text = []
        with open(file_path + file, 'r', encoding='latin-1') as obj:
            for line in obj:
                n_words = len(word_tokenize(line))
                if n_words in n_word_count:
                    n_word_count[n_words] += 1
                else:
                    n_word_count[n_words] = 1

                text.append(line)

        sentiment = np.ones([1, len(text)], dtype=int)        # For Positive File
        if file == neg_file:
            sentiment *= 0
        df = pd.concat([df, pd.DataFrame({'text': text, 'sentiment': sentiment[0]})], ignore_index=True)

    # Shuffling data
    df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    print('Word Frequency count ', n_word_count)

    return df





if __name__ == '__main__':
    load_movie_review()