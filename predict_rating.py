from keras.models import load_model
import utils

tokenizer = utils.load_tokens()
input_data = utils.convert_review('I like this a lot !', tokenizer)

model_name = 'model/model_amz_music_instruments.h5'
model = load_model(model_name)

predictions = model.predict(input_data)
utils.display_rating(predictions)