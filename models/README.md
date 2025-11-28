# Model Architecture

This document explains the LSTM-based Image Captioning model architecture.

---

## Overview

The model uses an **Encoder-Decoder architecture** that combines:
- **CNN Encoder**: Extracts visual features from images (InceptionV3)
- **LSTM Decoder**: Generates captions word by word

---

## Model Code

```python
from import_requirements import *

def build_model(vocab_size, max_caption_length, cnn_output_dim):
    # Image Feature Encoder
    input_image = Input(shape=(cnn_output_dim,), name='Features_Input')
    fe1 = BatchNormalization()(input_image)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)

    # Caption Sequence Encoder
    input_caption = Input(shape=(max_caption_length,), name="Sequence_Input")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input_caption)
    se2 = LSTM(256)(se1)

    # Decoder
    decoder1 = add([fe3, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax', name='Output_layer')(decoder2)

    model = Model(inputs=[input_image, input_caption], outputs=outputs, name="Image_captioning")
    return model
```

---

## Architecture Breakdown

### 1. Image Feature Encoder

Processes the visual features extracted by InceptionV3:

```python
input_image = Input(shape=(cnn_output_dim,), name='Features_Input')
fe1 = BatchNormalization()(input_image)
fe2 = Dense(256, activation='relu')(fe1)
fe3 = BatchNormalization()(fe2)
```

**Components:**
- **Input Layer**: Receives image features from InceptionV3
- **Batch Normalization**: Normalizes features for stable training
- **Dense Layer (256 units)**: Transforms features with ReLU activation
- **Batch Normalization**: Further stabilizes the feature representation

### 2. Caption Sequence Encoder

Processes the partial captions during training:

```python
input_caption = Input(shape=(max_caption_length,), name="Sequence_Input")
se1 = Embedding(vocab_size, 256, mask_zero=True)(input_caption)
se2 = LSTM(256)(se1)
```

**Components:**
- **Input Layer**: Receives tokenized caption sequences
- **Embedding Layer**: Converts word indices to dense vectors (256 dimensions)
  - `mask_zero=True`: Ignores padding tokens
- **LSTM Layer (256 units)**: Captures sequential dependencies in captions

### 3. Decoder

Combines visual and textual information to generate predictions:

```python
decoder1 = add([fe3, se2])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax', name='Output_layer')(decoder2)
```

**Components:**
- **Add Layer**: Merges image features and caption embeddings
- **Dense Layer (256 units)**: Processes the combined representation
- **Output Layer**: Produces probability distribution over vocabulary using softmax

---

## Model Inputs

The model accepts two inputs:

1. **Image Features** (`Features_Input`)
   - Shape: `(cnn_output_dim,)`
   - Pre-extracted features from InceptionV3

2. **Caption Sequence** (`Sequence_Input`)
   - Shape: `(max_caption_length,)`
   - Tokenized partial captions

---

## Model Output

**Probability Distribution** (`Output_layer`)
- Shape: `(vocab_size,)`
- Softmax probabilities for each word in the vocabulary
- Predicts the next word in the caption sequence

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| `vocab_size` | Total number of unique words in the vocabulary |
| `max_caption_length` | Maximum length of caption sequences |
| `cnn_output_dim` | Dimension of InceptionV3 output features (typically 2048) |

---

## Visual Representation

```
Image Features (2048)          Caption Sequence (max_length)
       ↓                                  ↓
 BatchNormalization                  Embedding (256)
       ↓                                  ↓
   Dense (256)                       LSTM (256)
       ↓                                  ↓
 BatchNormalization                       |
       ↓                                  |
       └──────────── Add ────────────────┘
                      ↓
                Dense (256, ReLU)
                      ↓
            Dense (vocab_size, Softmax)
                      ↓
              Next Word Prediction
```

---

## Key Features

- **Batch Normalization**: Improves training stability  
- **Embedding with Masking**: Handles variable-length captions  
- **LSTM**: Captures temporal dependencies in text  
- **Multimodal Fusion**: Combines visual and textual information  
- **Softmax Output**: Generates probability distribution for word prediction

---

## Model Summary

To view the complete model architecture:

```python
model = build_model(vocab_size=vocab_size, max_caption_length=34, cnn_output_dim=2048)
model.summary()
```

---

**Related Documentation:**
- [Training Guide](training/README.md)
- [Back to Main README](README.md)

---