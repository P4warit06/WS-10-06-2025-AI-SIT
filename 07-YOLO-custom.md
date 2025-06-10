### Train YOLO Model


pip install ultralytics


```python
from ultralytics import YOLO
```


```python
# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
```


```python
# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

results = model.train(
    data="stationery.v3i.yolov11/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    lr0=0.001,   # เริ่มต้น learning rate
    lrf=0.01     # learning rate ตอนสิ้นสุด (optional)
)

```


```python
# results = model(source="https://ultralytics.com/images/bus.jpg", show=True)  # predict and display results

results = model.predict(source="img-pen-pencil-max/561151_0.jpg", 
                               conf=0.3,
                               show=True)  # predict and display results
```


```python
%matplotlib inline
import cv2
import matplotlib.pyplot as plt

for result in results:
    image = result.plot()  # ได้ภาพใน BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

```


```python
# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
```


```python
results = model.predict(source="173684-849839047_tiny.mp4", show=True,conf=0.4, save=True)  # predict and save results
```


```python
model.names
```


```python

```
