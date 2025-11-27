import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
import pickle
from django.conf import settings
import os
from io import BytesIO

INFERENCE_DIR = os.path.join(settings.BASE_DIR, "inference_models")
MODEL_PATH = os.path.join(INFERENCE_DIR, "caption_model_v1.keras")
TOKENIZER_PATH = os.path.join(INFERENCE_DIR, "tokenizer_v1.pkl")


def preprocess_image(image_path):
  img = load_img(BytesIO(image_path.read()),target_size=(299,299))
  img = img_to_array(img)
  img = np.expand_dims(img,axis=0)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img

def extract_image_features(model,image_path):
  img = preprocess_image(image_path=image_path)
  features = model.predict(img,verbose=0)
  return features


caption_model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH,"rb") as f:
    tokenizer = pickle.load(f)


inception_v3_model = InceptionV3(weights='imagenet',include_top=False,pooling='avg')
cnn_ouput_dim = inception_v3_model.output_shape[1]
max_caption_length = 34
def caption_generator(image_features):
    in_text = 'start '
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length,padding='post').reshape((1,max_caption_length))
        prediction = caption_model.predict([image_features.reshape(1,cnn_ouput_dim), sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word[idx]
        in_text += ' ' + word
        if word == 'end':
            break

    in_text = in_text.replace('start ', '')
    in_text = in_text.replace(' end', '')

    return in_text
