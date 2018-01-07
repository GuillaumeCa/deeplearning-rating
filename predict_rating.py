# -*- coding: utf-8 -*-
from keras.models import load_model
import utils

tokenizer = utils.load_tokens()

review = "After a week only one side works"

input_data = utils.convert_review(review, tokenizer)

model_name = 'model/model_amz_1515341586.h5'
model = load_model(model_name)

predictions = model.predict(input_data)
utils.display_rating(predictions)