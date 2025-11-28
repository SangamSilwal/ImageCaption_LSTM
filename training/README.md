# Training Scripts

This directory contains scripts for training the Image Captioning LSTM model.

---

## Files

- `base.py` - Base configuration and constants
- `import_requirements.py` - Check required dependencies
- `model.py` - Model architecture and training script
- `models_utils.py` - Utility functions for data preprocessing
- `test.py` - Test the trained model

---

## Model Architecture

### Feature Extraction

I used the **InceptionV3** CNN pre-trained model for image feature extraction. The model is loaded without the top classification layer, using global average pooling to extract feature vectors from the last layer.

```python
inception_v3_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
```

### Optimizer Configuration

The model uses the **Adam optimizer** with the following configuration:
- Initial learning rate: `0.01`
- Gradient clipping: `clipnorm=1.0` (prevents exploding gradient problem)

```python
optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
```

### Loss Function

**Categorical Cross-Entropy** is used as the loss function, which is suitable for multi-class classification tasks like predicting the next word in a caption.

The formula for categorical cross-entropy is:

```
L = -Σ(yᵢ × log(ŷᵢ))
```

Where:
- **yᵢ** = True label (one-hot encoded)
- **ŷᵢ** = Predicted probability for class i
- **Σ** = Sum over all classes

This loss function penalizes incorrect predictions by measuring the difference between the true distribution and predicted distribution.

```python
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

---

## Training Strategy

### Preventing Overfitting

To prevent overfitting during training, I implemented **Early Stopping** with the following parameters:
- Monitors validation loss
- Patience: 3 epochs
- Restores best weights when training stops

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

### Learning Rate Scheduling

A **Learning Rate Scheduler** dynamically adjusts the learning rate during training to improve convergence:
- Keeps the initial learning rate for the first 3 epochs
- Reduces the learning rate by 10% (multiplies by 0.9) after epoch 3

```python
def lr_scheduler(epoch, lr):
    if epoch < 3:
        return lr
    return lr * 0.9

lr_schedule = LearningRateScheduler(lr_scheduler)
```

---

## Usage

### 1. Prepare Dataset

Place your dataset (Flickr8k, Flickr30k, or MS COCO) in the appropriate directory.
I have Flickr8k dataset. It is openly available in kaggle.

### 2. Configure Settings

Edit `base.py` to set dataset paths and hyperparameters.

### 3. Train Model

```bash
python base.py
```

### 4. Test Model

```bash
python test.py
```

---

## Output

Trained models will be saved in the `../models/` directory:
- `caption_model_v1.keras` - Trained LSTM model
- `tokenizer_v1.pkl` - Tokenizer for text preprocessing

---

**Related Documentation:**
- [Model Architecture](models/README.md)
- [Back to Main README](README.md)

---