import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ================= 데이터 경로 =================
train_dir = 'handwriting_dataset/'  # my_handwriting/ & other_handwriting/ 포함

# ================= 데이터 증강 =================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    color_mode='grayscale',
    batch_size=8,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    color_mode='grayscale',
    batch_size=8,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ================= CNN 모델 정의 =================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(optimizer=Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ================= 학습 =================
model.fit(train_gen, validation_data=val_gen, epochs=15)

# ================= 모델 저장 =================
model.save('my_handwriting_model.h5')
print("모델 학습 완료 및 저장됨!")