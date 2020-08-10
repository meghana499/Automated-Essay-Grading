from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from tensorflow.keras.models import Sequential, load_model, model_from_config
import tensorflow.keras.backend as K

def get_model():
    """Define the model."""
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


# #'''bidirectional lstm'''
# #from constants import GLOVE_DIR
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.models import Sequential
# from .utils import tokenizer, load_embedding_matrix

# def get_model(embedding_dimension, essay_length):
#     vocabulary_size = len(tokenizer.word_index) + 1
#     embedding_matrix = load_embedding_matrix(glove_directory=GLOVE_DIR, embedding_dimension=embedding_dimension)

#     model = Sequential()

#     model.add(Embedding(vocabulary_size, embedding_dimension, weights=[embedding_matrix], input_length=essay_length, trainable=False, mask_zero=False))
#     model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
#     model.add(Dropout(0.4))
#     model.add(LSTM(256, dropout=0.4, recurrent_dropout=0.4))
#     model.add(Dropout(0.4))
#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#     model.summary()

#     return model