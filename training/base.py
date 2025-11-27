from import_requirements import *
from models_utils import *
from model import build_model

# Images Directory and Caption file path
images_directory = './data/Images/'
captions_path = './data/captions.txt'

# Loading and Cleaning the captions
captions = load_captions(captions_path)
cleaned_captions = [clean_text(caption.split(',')[1]) for caption in captions]
captions_IDs = []
for i in range(len(cleaned_captions)):
  item = captions[i].split(',')[0] + '\t' + 'start ' + cleaned_captions[i] + ' end\n'
  captions_IDs.append(item)

# Tokenizing the Captions
tokenizer = tokenize_captions(cleaned_captions)
vocab_size = len(tokenizer.word_index)+1

# Preparing testing and validating dataset
all_images_ids = os.listdir(images_directory)
train_image_ids,val_image_ids = train_test_split(all_images_ids,test_size=0.15,random_state=42)
val_image_ids, test_image_ids = train_test_split(val_image_ids,test_size=0.1,random_state=42)
train_captions, val_captions,test_captions = [],[],[]
for caption in captions_IDs:
  image_id, _ = caption.split('\t')
  if image_id in train_image_ids:
    train_captions.append(caption)
  elif image_id in val_image_ids:
    val_captions.append(caption)
  elif image_id in test_image_ids:
    test_captions.append(caption)
  else:
    print("Unknown image ID! ")


# Extracting all the image features from the pretrained InceptionV3 model
inception_v3_model = InceptionV3(weights='imagenet',include_top=False,pooling='avg')
train_image_features, val_image_features,test_image_features = {},{},{}
pbar = tqdm(total=len(all_images_ids),position=0,leave=True,colour='green')
for caption in all_images_ids:
  image_id = caption.split('\t')[0]
  image_path = os.path.join(images_directory,image_id)
  image_features = extract_image_features(inception_v3_model,image_path)
  if image_id in train_image_ids:
    train_image_features[image_id] = image_features.flatten()
    pbar.update(1)
  elif image_id in val_image_ids:
    val_image_features[image_id] = image_features.flatten()
    pbar.update(1)
  elif image_id in test_image_ids:
    test_image_features[image_id] = image_features.flatten()
    pbar.update(1)
  else:
    print("Unknown Image ID! ")
pbar.close()


# yielding the X_images and X_captions which will be set according to the setps_per_epoch
def data_generator(captions,image_features,tokenizer,max_caption_length,batch_size):
  num_samples = len(captions)
  image_ids = list(image_features.keys())
  while True:
    np.random.shuffle(image_ids)

    for start_idx in range(0,num_samples,batch_size):
      end_idx = min(start_idx + batch_size,num_samples)
      X_images,X_captions,y = [],[],[]
      for caption in captions[start_idx:end_idx]:
        image_id,caption_text = caption.split('\t')
        caption_text = caption_text.rstrip('\n')
        seq = tokenizer.texts_to_sequences([caption_text])[0]

        for i in range(1,len(seq)):
          in_seq , out_seq = seq[:i],seq[i]
          in_seq = pad_sequences([in_seq],maxlen=max_caption_length,padding='post')[0]
          out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
          X_images.append(image_features[image_id])
          X_captions.append(in_seq)
          y.append(out_seq)
      yield (np.array(X_images),np.array(X_captions)),np.array(y)

max_caption_length = max(len(caption.split()) for caption in cleaned_captions)+1
cnn_ouput_dim = inception_v3_model.output_shape[1]
batch_size_train = 270
batch_size_val = 150
train_data_generator = data_generator(train_captions,train_image_features,tokenizer,max_caption_length,batch_size_train)
val_data_generator = data_generator(val_captions,val_image_features,tokenizer,max_caption_length,batch_size_val)


# Bulding Model
caption_model = build_model(vocab_size,max_caption_length,cnn_ouput_dim)
optimizer = Adam(learning_rate=0.01,clipnorm=1.0)  #Taking the clipnorm =1.0 to prevent from exploding gradient descent problem
caption_model.compile(loss='categorical_crossentropy',optimizer=optimizer)


# Training the model
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
lr_schedule = LearningRateScheduler(lr_scheduler)

history = caption_model.fit(
    train_data_generator,
    steps_per_epoch=math.ceil(len(train_captions) / batch_size_train),
    validation_data=val_data_generator,
    validation_steps=math.ceil(len(val_captions) / batch_size_val),
    epochs=15,
    callbacks=[early_stopping, lr_schedule]
)


# Saving the models
caption_model.save("caption_model_v1.keras")
print("Model Saved")
with open('tokenizer_v1.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)