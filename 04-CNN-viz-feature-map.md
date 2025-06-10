### Explore CNN Feature Map


```python
from tensorflow.keras.models import load_model

model = load_model("my_model.keras")  # or "my_model.h5"

```


```python
model.layers 
```


```python
vgg_base = model.layers[0]  # คือ Functional VGG16 model
for i, layer in enumerate(vgg_base.layers):
    print(i, layer.name, layer.output.shape)

```


```python
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt

# เลือกเลเยอร์ใน VGG16 ที่ต้องการดู เช่น block3_conv2
target_layer = 'block3_conv2'
vgg_base = model.layers[0]

# สร้างโมเดลย่อยที่รับภาพแล้วคืนค่า activation ของเลเยอร์ที่ต้องการ
activation_model = Model(inputs=vgg_base.input,
                         outputs=vgg_base.get_layer(target_layer).output)

# เตรียมภาพ
img = cv2.imread("D:/MLpython310/ML-Image/Dataset/test/hand-zero/20210711-203645.jpg")
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = preprocess_input(np.expand_dims(img, axis=0))

# ทำนาย
activation = activation_model.predict(img)

# แสดง Feature Map
plt.figure(figsize=(12, 12))
for i in range(16):  # แสดงแค่ 16 ฟิลเตอร์แรก
    plt.subplot(4, 4, i + 1)
    plt.imshow(activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle(f'Feature Maps from {target_layer}')
plt.show()

```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 151ms/step
    


    
![png](04-CNN-viz-feature-map_files/04-CNN-viz-feature-map_4_1.png)
    

