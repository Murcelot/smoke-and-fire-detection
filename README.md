Models for detection smoke and fire on custom dataset.

In this work were compared 2 models: resnet-50-detr and YOLO11x for object detection task.
Weights for both models can be found here:
https://disk.yandex.ru/d/hGic6FWNg8GmRQ
unzip it directly to 'weights' for proper work.

Dataset can be found here:
https://disk.yandex.ru/d/b8-Nk96jYO7hJg
unzip it directly to 'data' for proper work.

'preparing_data.ipynb' has code for preparing data in required format.
'detr_train.ipynb' has code for downstream task training resnet-50-detr.
YOLO11x was trained using ultralytics CLI, config for training can be found in 'weights' - 'args.yaml'

Both models were pretrained on COCO
Resulting metrics of downstream training presented in table:

| Model       | mAP    | mAP@50 | mAP@75 | mAP (small) | mAP (medium) | mAP (large) | mAP_smoke | mAP_fire | MAR@1   | MAR@10  | MAR@100 | MAR (small) | MAR (medium) | MAR (large) | mAR_100_smoke | mAR_100_fire |
|--------------|--------|--------|--------|-------------|--------------|-------------|-----------|----------|---------|---------|---------|-------------|--------------|-------------|---------------|--------------|
| YOLO11x      | 0.2414 | 0.4746 | 0.2200 | 0.0883      | 0.2467       | 0.3184      | 0.1755    | 0.3074   | 0.2006  | 0.3686  | 0.4513  | 0.2673      | 0.4505       | 0.5194      | 0.4013        | 0.5012       |
| Model 2      | —      | —      | —      | —           | —            | —           | —         | —        | —       | —       | —       | —           | —            | —           | —             | —            |
