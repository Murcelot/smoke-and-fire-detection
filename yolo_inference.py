from ultralytics import YOLO
import os
# Load a model
model_path = os.path.join(os.getcwd(), 'weights', 'yolo_weights', 'best.pt')
model = YOLO(model_path)  # pretrained YOLO11n model

id2label = {0 : 'smoke', 1 : 'fire'}
# Input image
print('Input path to image to detect smoke and fire: ')
img_path = input()

# Run batched inference on a list of images
results = model([img_path], verbose=False)  # return a list of Results objects

# Process results list
for result in results:
    bbox_res = result.boxes
    for score, label, box in zip(bbox_res.conf, bbox_res.cls, bbox_res.xyxy):
        box = [round(i, 2) for i in box.tolist()]
        # print(score)
        # print(label)
        # print(box)
        print(
            f"Detected {id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    result.save(filename = img_path[:-4] + '_detected_YOLO' + img_path[-4:])  # save to disk

print('Image with bboxes saved in ' + img_path[:-4] + '_detected_YOLO' + img_path[-4:])