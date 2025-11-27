from models_utils import extract_image_features
from import_requirements import *
import pickle



caption_model = load_model('./models/caption_model_v1.keras')

with open("./models/tokenizer_v1.pkl","rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer Loaded")




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


image_test1 = extract_image_features(inception_v3_model,"test_5.webp")
cap = caption_generator(image_test1)
print(cap)