import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from transformers.utils import logging

logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

def convert_to_rgb(image):
    return image.convert('RGB')

image_path = 'Food Pictures/food_1.png'
image_raw = Image.open(image_path)
image = convert_to_rgb(image_raw)

plt.imshow(image)
plt.axis('off')
plt.show()

modelId = "Kaludi/food-category-classification-v2.0"

model = pipeline("image-classification", model=modelId)

class_scores = model(image)
print(class_scores)

highest_probability = max(class_scores, key=lambda x: x['score'])
highest_probability_class = highest_probability['label']

print(f"Predicted food category: {highest_probability_class}")
