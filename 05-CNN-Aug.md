### CNN


### Gen Augment Train Images


```python
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

src_dir = 'D:/MLpython310/ML-Image/Dataset/train/'
dst_dir = 'D:/MLpython310/ML-Image/Dataset/train_augmented/'

# ล้างโฟลเดอร์เก่า ถ้ามี
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

total_per_class = 100

for cls in classes:
    print(f"Augmenting class: {cls}")
    src_cls_dir = os.path.join(src_dir, cls)
    dst_cls_dir = os.path.join(dst_dir, cls)
    os.makedirs(dst_cls_dir, exist_ok=True)

    # สร้าง generator แยกสำหรับแต่ละคลาส
    generator = datagen.flow_from_directory(
        directory=src_dir,
        target_size=(224, 224),
        classes=[cls],               # เลือกคลาสนี้เท่านั้น
        batch_size=16,
        class_mode=None,
        shuffle=True,
        save_to_dir=dst_cls_dir,    # บันทึกในโฟลเดอร์คลาสนี้
        save_prefix='aug',
        save_format='jpg'
    )

    generated = 0
    while generated < total_per_class:
        batch = next(generator)
        generated += len(batch)
        print(f"  {cls}: Generated {generated}/{total_per_class}")

```


```python
import tensorflow as tf
from tensorflow import keras

train_ds_1 = keras.utils.image_dataset_from_directory(
    directory='D:/MLpython310/ML-Image/Dataset/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(224, 224),
    shuffle=True,  # shuffle ในแต่ละ dataset
    seed=123
)

train_ds_2 = keras.utils.image_dataset_from_directory(
    directory='D:/MLpython310/ML-Image/Dataset/train_augmented/',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(224, 224),
    shuffle=True,
    seed=123
)

# รวม dataset ทั้งสอง
train_ds = train_ds_1.concatenate(train_ds_2)

# shuffle dataset หลังรวม (buffer size เลือกตามจำนวนข้อมูลทั้งหมด หรือใหญ่ๆ)
train_ds = train_ds.shuffle(buffer_size=1000, seed=123)


test_ds = keras.utils.image_dataset_from_directory(
    directory='D:/MLpython310/ML-Image/Dataset/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(224, 224))

```


```python
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers

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

```
