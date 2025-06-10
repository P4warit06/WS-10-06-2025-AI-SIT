### Load and Apply Model to Predict 


```python
from ultralytics import YOLO
```


```python
# Load a COCO-pretrained YOLO11n model
model = YOLO("runs/detect/train73/weights/best.pt")
```


```python
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


    
![png](08-YOLO-load-bst-model_files/08-YOLO-load-bst-model_4_0.png)
    



```python
results = model(source="pen-pencil-test.mp4", show=True,conf=0.4, save=True)  # predict and save results
```


```python

```
