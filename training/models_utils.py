from import_requirements import *


def load_captions(file_path):
    with open(file_path,'r') as f:
        captions = f.readlines()
        captions = [caption.lower() for caption in captions[1:]]
    return captions

def tokenize_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_image(image_path):
  img = load_img(image_path,target_size=(299,299))
  img = img_to_array(img)
  img = np.expand_dims(img,axis=0)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img

def extract_image_features(model,image_path):
  img = preprocess_image(image_path=image_path)
  features = model.predict(img,verbose=0)
  return features

def lr_scheduler(epoch, lr):
    if epoch < 3:
        return lr
    return lr * 0.9