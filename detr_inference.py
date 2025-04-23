import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from accelerate.test_utils.testing import get_backend
import os

import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# Loading model and get device
id2label = {0 : 'Smoke', 1 : 'Fire'}

device, _, _ = get_backend()
model_path = os.path.join(os.getcwd(), 'weights', 'detr_weights')
image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
model = AutoModelForObjectDetection.from_pretrained(model_path)
model = model.to(device)

print('Input path to image to detect smoke and fire: ')
img_path = input()

image = Image.open(img_path)

# Make predictions
with torch.no_grad():
    inputs = image_processor(images=[image], return_tensors="pt")
    outputs = model(**inputs.to(device))
    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

# Print detected items
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

# Make result image
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    if label.item == 0:
        draw.rectangle((x, y, x2, y2), outline="red", width=5)
    else:
        draw.rectangle((x, y, x2, y2), outline="green", width=5)

print('Smoke in red bbox, fire in green bbox')
image.save(img_path[:-4] + '_detected' + img_path[-4:])
