# -*- coding: utf-8 -*-
from keras.models import load_model
import utils

tokenizer = utils.load_tokens()

review = "I'm a pro-cheapo and I hated this thing. They're noisy, and the cables feel really cheap, gummy-like. Drop few more bucks and get something else!"

input_data = utils.convert_review(review, tokenizer)

model_name = 'model/model_amz_music_instruments.h5'
model = load_model(model_name)

predictions = model.predict(input_data)
utils.display_rating(predictions)