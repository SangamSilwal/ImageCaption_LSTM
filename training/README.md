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

## Usage

### 1. Prepare Dataset

Place your dataset (Flickr8k/Flickr30k/MS COCO) in the appropriate directory.

### 2. Configure Settings

Edit `base.py` to set dataset paths and hyperparameters.

### 3. Train Model
```bash
python model.py
```

### 4. Test Model
```bash
python test.py
```

---

## Output

Trained models will be saved in `../models/` directory:
- `caption_model_v1.keras`
- `tokenizer_v1.pkl`

---