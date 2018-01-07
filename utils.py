import numpy as np
import pickle
from keras.preprocessing import sequence

def ratingToBitArr(rate):
    arr = np.zeros(5, dtype=int)
    arr[rate - 1] = 1
    return list(arr)

def save_tokens(tokenizer):
    with open('model/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokens():
    with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer

def convert_review(review, tokenizer):
    sequences = tokenizer.texts_to_sequences([review])
    return sequence.pad_sequences(sequences, maxlen=500)

def display_rating(rating_vector):
    ratings = [1, 2, 3, 4, 5]
    threshold = 0.6
    for index, val in enumerate(rating_vector[0]):
        print 'Rate: {} \t predict: {:.2f}% {}'.format(
            ratings[index], 
            val * 100, '*' if val > threshold else ''
        )