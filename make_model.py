import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ==================== 2. 데이터 경로 설정 ====================
train_dir = "handwriting_dataset_resized"

# ==================== 3. 데이터 증강 및 전처리 ====================
datagen = ImageDataGenerator(
    rescale=1./255,       # 픽셀값을 0~255 → 0~1 사이로 정규화
    validation_split=0.2  # 전체 데이터 중 20%는 검증용(validation)으로 사용
)

# 학습용 데이터 불러오기
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    color_mode='grayscale',
    batch_size=8,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# 검증용 데이터 불러오기
val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    color_mode='grayscale',
    batch_size=8,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ==================== 4. CNN 모델 만들기 ====================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ==================== 5. 모델 컴파일 ====================
model.compile(optimizer=Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ==================== 6. 모델 학습 ====================
model.fit(train_gen, validation_data=val_gen, epochs=15)

# ==================== 7. 모델 저장 ====================
model.save("my_handwriting_model.h5")