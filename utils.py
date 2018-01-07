import numpy as np

def ratingToBitArr(rate):
    arr = np.zeros(5, dtype=int)
    arr[rate - 1] = 1
    return list(arr)