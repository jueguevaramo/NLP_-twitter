# %% md
# Main

# %%

import pandas as pd
import tensorflow as tf
from Tokanizer import TOC
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

path = '/home/jueguevaramo/Documents/Data_sets/Kaggle/'
path = path + 'Tweet_NLP/tweet-sentiment-extraction/train.csv'
data = pd.read_csv(path)

data['text'] = data['text'].astype('str')
data['selected_text'] = data['selected_text'].astype('str')

data['text_seq'] = data.text.apply(lambda x: TOC(x).from_tweet())
data['selected_text_seq'] = data.selected_text.apply(lambda x:
                                                     TOC(x).from_tweet())
data['len_text'] = data.text_seq.apply(lambda x: len(x))
data['len_selected_text'] = data.selected_text_seq.apply(lambda x: len(x))
data['cant'] = data.len_selected_text/data.len_text


# %%
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

X_train, X_test, y_train, y_test = train_test_split(data.text,
                                                    data.selected_text,
                                                    test_size=0.4)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

training_sequences = tokenizer.texts_to_sequences(X_train)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


def sequence_to_text(list_of_indices, reverse_word_map):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)


data.sample(n=5)
# %%
