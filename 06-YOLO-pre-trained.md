pip install ultralytics

pip install yt-dlp

yt-dlp https://www.youtube.com/shorts/hHCw0rzicIY -o myvideo.mp4 



```python
from ultralytics import YOLO
```


```python
# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
```


```python
# results = model(source="https://ultralytics.com/images/bus.jpg", show=True, save=True)  # predict and display results

# results = model(source="images-butterfly.jpg", show=True, save=True)  # predict and display results

results = model(source="img-pen-pencil-max/561151_0.jpg", show=True, save=True)  # predict and display results
```

    

    


```python
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
results = model.predict(source="myvideo.mp4", show=True,conf=0.4, save=True) #, project="result-detect")  # predict and save results
```


```python

```
