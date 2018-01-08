import numpy

# from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.preprocessing import sequence, text
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from matplotlib import pyplot
import json
import time

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

# Create the model
model = Sequential()
model.add(Embedding(TOP_WORDS, EMBEDDING_VECTOR_LENGTH, input_length=MAX_REVIEW_LENGTH))
model.add(LSTM(MEMORY_UNITS))
model.add(Dropout(DROPOUT))
model.add(Dense(5, activation='softmax'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print model.summary()

# filepath="model/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [TensorBoard(), checkpoint]
callbacks_list = []

history = model.fit(
    X_train, 
    y_train, 
    validation_data=(X_test, y_test), 
    epochs=NB_EPOCHS, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    callbacks=callbacks_list)

scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')

pyplot.figure()
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

if SAVE_MODEL:
    print 'saving model...'
    model.save('model/model_amz_' + str(time.time())[:-3] + '.h5')
