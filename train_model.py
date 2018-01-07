import numpy as np

# from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
import json

import utils

# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

# config params
max_review_length = 500
top_words = 5000
reviewTextKey = 'reviewData'
embedding_vector_length = 32
memory_units = 100
nb_epochs = 3
batch_size = 64

save_model = False

# import dataset
print 'importing data...'
dataset = open('dataset/reviews_Musical_Instruments_5.json', 'r').readlines()

reviewsText = []
reviewsRatings = []

tokenizer = text.Tokenizer(num_words=top_words)

# format data
print 'formating data...'
for rawline in dataset:
  review = json.loads(rawline.strip())
  reviewsRatings += [utils.ratingToBitArr(int(review['overall']))]
  reviewsText += [review['reviewText']]

# learning tokens
tokenizer.fit_on_texts(reviewsText)

# saving tokens to predict reviews rating in the future
utils.save_tokens(tokenizer)

# convert sequences of words to sequences of integers
reviewsText = tokenizer.texts_to_sequences(reviewsText)

# Create training and testing sets
train_selection = int(len(reviewsText) / 2.0)

X_train = reviewsText[:train_selection]
y_train = reviewsRatings[:train_selection]

X_test = reviewsText[train_selection:]
y_test = reviewsRatings[train_selection:]

# Pad input text
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Create the model
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(memory_units))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epochs, batch_size=batch_size)

scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)

if save_model:
    model.save('model/model_amz_music_instruments.h5')
