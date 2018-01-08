# -*- coding: utf-8 -*-
from keras.models import load_model
import utils

tokenizer = utils.load_tokens()

review = "Good but the charger is not the same size as an Apple charger and that makes it a little difficult for charging with cases on." # 2

input_data = utils.convert_review(review, tokenizer)

model_name = 'model/weights-improvement-20-0.9417.hdf5'
model = load_model(model_name)

predictions = model.predict(input_data)
utils.display_rating(predictions)