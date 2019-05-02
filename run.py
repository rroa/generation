from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import pandas as pd
import numpy as np
import string, os 
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


essay_df = pd.read_csv('data.csv', encoding="latin-1")
all_descriptions = list(essay_df.desc.values)

#colnames = ['description']
#data = pd.read_csv('data.csv', names=colnames)
#all_description = data.description.tolist()

len(all_descriptions)

corpus = [x for x in all_descriptions]
corpus[:1]

t = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
t.fit_on_texts(corpus)


# A dictionary of words and their counts.
print(t.word_counts)

# A dictionary of words and how many documents each appeared in.
print(t.word_docs)

# An integer count of the total number of documents that were used to fit the Tokenizer (i.e. total number of documents)
print(t.document_count)

# A dictionary of words and their uniquely assigned integers.
print(t.word_index)

print('Found %s unique tokens.' % len(t.word_index))


# Tokenization
t = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)

def get_sequence_of_tokens(corpus):
    t.fit_on_texts(corpus)
    total_words = len(t.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = t.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words
input_sequences, total_words = get_sequence_of_tokens(corpus)

input_sequences[:10]

# pad sequences
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_len, padding = 'pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes = total_words)

    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(input_sequences)

def create_model(max_sequence_len, total_words):
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

model = create_model(max_sequence_len, total_words)
model.summary()

model.fit(predictors, label, epochs=100, verbose=5)

def generate_text(seed_text, next_words, model, max_seq_len):
    for _ in range(next_words):
        token_list = t.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ''

        for word,index in t.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text = seed_text + " " + output_word

    return seed_text.title()


print(generate_text("proudest accomplished moment", 100, model, max_sequence_len))
print()
print(generate_text("established proudest appreciated workflow accomplished", 200, model, max_sequence_len))
print()
print(generate_text('observed established proudest appreciated workflow accomplished', 300, model, max_sequence_len))
