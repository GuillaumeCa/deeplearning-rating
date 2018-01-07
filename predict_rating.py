from keras.models import load_model
import utils

model_name = 'model_amz_music_instruments.h5'
model = load_model(model_name)

model.predict()