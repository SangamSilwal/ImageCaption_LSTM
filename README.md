# Image Captioning LSTM

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2.8-green)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.12.0-red)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.5-lightgrey)](https://numpy.org/)

---

## Description

This project is an **Image Captioning System** using **LSTM (Long Short-Term Memory)** networks.  
Users can upload an image, and the system generates a relevant caption describing the image.  

The backend is built using **Django**, and the model is trained using **TensorFlow/Keras**.

---

## Root Directory

```
project-root/
├── core/
├── models/
├── training/
|── README.md
|__requirements.txt

```

---

## Core Directory

The `core/` directory contains the Django application and configuration files.

```
core/
├── app/
│   ├── templates/
│   │   └── index.html
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── utils.py
│   ├── views.py
│   └── __init__.py
├── core/
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── __init__.py
├── inference_models/
│   ├── caption_model_v1.keras
│   └── tokenizer_v1.pkl
├── db.sqlite3
└── manage.py
```

---

## Models Directory

The `models/` directory stores trained model files.

```
models/
├── caption_model_v1.keras
└── tokenizer_v1.pkl
```

### Breakdown

- `caption_model_v1.keras`: Trained Keras model for image captioning
- `tokenizer_v1.pkl`: Pickled tokenizer object for text processing

---

## Training Directory

The `training/` directory contains scripts for model training and testing.

```
training/
├── base.py
├── import_requirements.py
├── model.py
├── models_utils.py
└── test.py
```

### Breakdown

- `base.py`: Base configuration and constants
- `import_requirements.py`: Script to check and import required dependencies
- `model.py`: Model architecture definition
- `models_utils.py`: Utility functions for model training (data loading, preprocessing, etc.)
- `test.py`: Script to test the trained model

---

## Notes

- The project uses **Django** for the web interface
- **TensorFlow/Keras** models are stored in both `core/inference_models/` (for deployment) and `models/` (for development)
- Training scripts are separated from the Django application for modularity

---
# Installation Guide

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/SangamSilwal/ImageCaption_LSTM.git
```

### 2. Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
cd app
cd core
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your browser.

---
## Screenshots

![Home Page](https://private-user-images.githubusercontent.com/181588051/519970654-36ae5bbf-4178-43fe-98a9-ef4604731bec.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjQzMDYzNTQsIm5iZiI6MTc2NDMwNjA1NCwicGF0aCI6Ii8xODE1ODgwNTEvNTE5OTcwNjU0LTM2YWU1YmJmLTQxNzgtNDNmZS05OGE5LWVmNDYwNDczMWJlYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMTI4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTEyOFQwNTAwNTRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kYzRmM2YzODAzOTM3ZTBjNDhkNmExMWZmMjQyZDMzYjYzNzM2MDFjNDM3MjBmNDc2ZTY0Yjg5Y2IzYmE1NjA5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.BWrLhq3TP3WdfxrpvRtS2kls1A_0FviDRsVjqBQEMnw)

![Generated caption](https://private-user-images.githubusercontent.com/181588051/519971739-91ab4cca-8506-4b5f-b618-c8b88f723e64.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjQzMDY2MTksIm5iZiI6MTc2NDMwNjMxOSwicGF0aCI6Ii8xODE1ODgwNTEvNTE5OTcxNzM5LTkxYWI0Y2NhLTg1MDYtNGI1Zi1iNjE4LWM4Yjg4ZjcyM2U2NC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMTI4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTEyOFQwNTA1MTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jNDA2M2Y2NzJkMTQwMDg5NmNiODk3Mzg2Y2NiNzYwNzk3MDdkMjg3MTRkMmU4ZTUxZDIwYjYzZWRmZWUyMDAyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.bgu_WQq1-9dSh8QvSh3u210A2twxiaENZl4YOrI6RBQ)