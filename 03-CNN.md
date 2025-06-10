### CNN



```python
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers

# Load the dataset
train_ds = keras.utils.image_dataset_from_directory(
    directory='D:/MLpython310/ML-Image/Dataset/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(224, 224))
test_ds = keras.utils.image_dataset_from_directory(
    directory='D:/MLpython310/ML-Image/Dataset/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(224, 224))
```


```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  #  have 2 classes
])
model.summary()
```


```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=20, validation_data=test_ds)
```


```python
import matplotlib.pyplot as plt
# Plot training history for accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training history for loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## CNN pre-trained

1. เตรียมข้อมูล train และ test ตามโฟลเดอร์ ที่เก็บข้อมูลไว้


```python
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers

train_ds = keras.utils.image_dataset_from_directory(
    directory='D:/MLpython310/ML-Image/Dataset/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))
test_ds = keras.utils.image_dataset_from_directory(
    directory='D:/MLpython310/ML-Image/Dataset/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))
```

2. แสดงชื่อคลาส


```python
class_names = train_ds.class_names
print(class_names)
```


```python
import json

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

```

3. สร้างชั้น CNN  โดยใช้ package VGG16 กำหนด ดังนี้ 
- ไม่เอาชั้น fully connected nn โดยกำหนด include_top=False
- weights='imagenet'
- input_shape=(224,224,3)





```python
#Exam2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16  #https://keras.io/api/applications/vgg/
import numpy as np

base = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
base.trainable = False

model = keras.Sequential()
model.add(base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary() 
```

4. แสดงโครงสร้างของ base model ในข้อ 3



```python
base.summary()
```

5. กำหนดให้ไม่ต้องมีการปรับค่า weight ของ base model


```python
base.trainable = False
```

6. กำหนดโครงสร้างของโมเดล สำหรับสอนข้อมูลภาพที่เราสร้างไว้ 


```python
model = keras.Sequential()
model.add(base)
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
model.summary() 
```

7. กำหนดรูปแบบ การเรียนรู้ 
- loss='categorical_crossentropy' 
- optimizer='adam'
- metrics=['accuracy']


```python
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
```

8. ให้โมเดลเรียนรู้จากข้อมูลสอน จำนวน 10 รอบ และกำหนด validation_data=test_ds


```python
# model.fit(train_ds, epochs=10, validation_data=test_ds)

# Train the model
history = model.fit(train_ds, epochs=20, validation_data=test_ds)
```


```python
import matplotlib.pyplot as plt
# Plot training history for accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training history for loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```


```python
model.summary()
```

9. บันทึก model ชื่อ 6xxxxxxx_model.h5


```python
# model.save('6xxxxxxx_model.h5')
model.save("my_model.keras")  # or .h5

```

10. อ่านภาพ และทำนาย ภาพ ว่าอยู่คลาสใด


```python
from tensorflow.keras.models import load_model

model = load_model("my_model.keras")  # or "my_model.h5"

```


```python
import json
with open("class_names.json", "r") as f:
    class_names = json.load(f)

```


```python
import cv2
import tensorflow as tf
import numpy as np

im = cv2.imread('D:/MLpython310/ML-Image/Dataset/test/hand-zero/20210711-203645.jpg')
imrz = cv2.resize(im,(224,224))
im_arr = tf.keras.utils.img_to_array(imrz)
im_arr = np.array([im_arr])  # Convert single image to a batch.
predictions = model.predict(im_arr)
str_class = class_names[predictions.argmax()]
print('Class: ', str_class)

image = cv2.putText(im,str_class,(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("image", image)
cv2.waitKey()


```


```python
cv2.destroyAllWindows()
```


```python
import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    label = f"{class_names[class_index]}: {confidence:.2f}"

    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow("Webcam Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
