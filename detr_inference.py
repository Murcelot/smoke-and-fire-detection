import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from accelerate.test_utils.testing import get_backend
import os

import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


device, _, _ = get_backend()
model_path = os.path.join(os.getcwd(), 'weights', 'detr_weights')

image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
model = AutoModelForObjectDetection.from_pretrained(model_path)
model = model.to(device)

print('Input path to image to detect smoke and fire: ')
img_path = input()

image = Image.open(img_path)

print('Done!')
