from django.shortcuts import render
from .utils import extract_image_features,caption_generator,inception_v3_model


def index(request):
    caption = None

    if request.method == "POST" and request.FILES.get('image'):
        image_file = request.FILES["image"]
        image_features = extract_image_features(inception_v3_model,image_file)
        caption = caption_generator(image_features)

    return render(request,"index.html",{"caption":caption})