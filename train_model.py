import numpy

# from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
import json

import utils
from config import *


numpy.random.seed(7)

# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

# import dataset
print 'importing data...'
dataset = open('dataset/' + DATASET_PATH, 'r').readlines()

reviewsText = []
reviewsRatings = []

tokenizer = text.Tokenizer(num_words=TOP_WORDS)

ratingsStats = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}

# format data
print 'formating data...'
for rawline in dataset:
    review = json.loads(rawline.strip())
    rate = int(review['rating'])
    if ratingsStats[rate] < 5000:
        ratingsStats[rate] += 1
        reviewsRatings += [rate - 1]
        reviewsText += [review['text']]

# learning tokens
tokenizer.fit_on_texts(reviewsText)

word_index = tokenizer.word_index
print 'found {} unique tokens.'.format(len(word_index))

# saving tokens to predict reviews rating in the future
utils.save_tokens(tokenizer)

# convert sequences of words to sequences of integers
reviewsText = tokenizer.texts_to_sequences(reviewsText)

print reviewsRatings[:10]
reviewsRatings = to_categorical(reviewsRatings, num_classes=5)

# Create training and testing sets
train_selection = int(len(reviewsText) / 2.0)

X_train = reviewsText[:train_selection]
y_train = reviewsRatings[:train_selection]
X_test = reviewsText[train_selection:]
y_test = reviewsRatings[train_selection:]

# Pad input text
X_train = sequence.pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)

# print 'train: ', X_train[:1], len(X_train[0])
# print 'test: ', X_test[:1], len(X_test[0])

# Create the model
model = Sequential()
model.add(Embedding(TOP_WORDS, EMBEDDING_VECTOR_LENGTH, input_length=MAX_REVIEW_LENGTH))
model.add(LSTM(MEMORY_UNITS, dropout=DROPOUT))
model.add(Dense(5, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=NB_EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

if SAVE_MODEL:
    print 'saving model...'
    model.save('model/model_amz_music_instruments.h5')
