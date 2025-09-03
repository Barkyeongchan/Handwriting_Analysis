# 필적 감정 프로그램 (Handwriting Analysis)

## 목차
**1. 프로젝트 설명**
- 목표
- 개발 환경

**2. 작동 과정**
- 샘플 이미지 패딩
- 모델 학습

**3. 결과 예측**

## 1. 프로젝트 설명

### **[목표]**

나의 글씨체와 다른 사람의 글씨체를 학습시킨 데이터셋를 활용하여 필적 감정을 완수한다.

### **[개발 환경]**

`운영체제(OS)` : Windows 10

`개발 언어` : Python 3.10

`패키지 및 라이브러리` : tensorflow 2.20 / keras 3.11 / numpy 2.2 / opencv-python 4.12

`IDE / 에디터` : VS Code

[HuggingFace](https://huggingface.co/spaces/parkyc/Handwriting_Analysis)

## 2. 작동 과정

### 2-1. 샘플 이미지 패딩(Padding)

`이미지 패딩` : 이미지를 원하는 크기/비율로 맞추기 위해 **빈 공간을 추가**하는 과정

<img width="354" height="295" alt="image" src="https://github.com/user-attachments/assets/666beee6-fe0c-42be-a5d2-09ac8a0d9d05" />

<img width="354" height="295" alt="image" src="https://github.com/user-attachments/assets/dbc38c40-6b2a-4b10-8155-55c5185cf54c" />

<br><br>

<img width="354" height="295" alt="image" src="https://github.com/user-attachments/assets/430989d9-00f2-453c-aae4-ad9e696651d3" />

<img width="354" height="295" alt="image" src="https://github.com/user-attachments/assets/e9a3d275-6311-459e-85c9-84b1d17f8827" />

```python
import cv2
import os

input_dir = "handwriting_dataset"  # 패딩 하기 전 폴더
output_dir = "handwriting_dataset_resized"  # 패딩 한 후 폴더
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 자동 생성

# 최종적으로 이미지를 맞출 크기 (정사각형으로 통일)
target_size = 128  

# ===================== 함수 정의 =====================
def resize_with_padding(img, target_size=128):
    """
    이미지를 비율을 유지하면서 축소/확대하고,
    남는 부분은 검은색 패딩으로 채워서 정사각형으로 만드는 함수
    """

    h, w = img.shape[:2]  # 이미지의 세로(height), 가로(width) 크기
    scale = target_size / max(h, w)  # 긴 쪽 기준으로 줄이는 비율 계산
    new_w, new_h = int(w * scale), int(h * scale)  # 리사이즈 후 가로, 세로 크기

    # 이미지 크기 변경 (비율 유지한 채로)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 최종 캔버스(검은색 배경) 만들고 이미지 붙이기
    canvas = cv2.copyMakeBorder(
        resized,
        top=(target_size - new_h) // 2,               # 위쪽 패딩
        bottom=(target_size - new_h + 1) // 2,        # 아래쪽 패딩
        left=(target_size - new_w) // 2,              # 왼쪽 패딩
        right=(target_size - new_w + 1) // 2,         # 오른쪽 패딩
        borderType=cv2.BORDER_CONSTANT,               # 일정한 색으로 채움
        value=(0, 0, 0)                               # 검은색 (RGB 0,0,0)
    )
    return canvas

# 폴더 안에 있는 모든 이미지 파일 하나씩 처리
for root, _, files in os.walk(input_dir):  # 하위 폴더까지 탐색
    for file in files:
        # 이미지 파일만 선택 (jpg, jpeg, png)
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)  # 이미지 경로
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 흑백으로 불러오기

            processed = resize_with_padding(img, target_size)  # 리사이즈+패딩 처리

            # 저장할 위치 (원래 폴더 구조 유지하면서 새 폴더에 저장)
            rel_path = os.path.relpath(root, input_dir)
            save_dir = os.path.join(output_dir, rel_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file)

            cv2.imwrite(save_path, processed)  # 변환된 이미지 저장
```

### 2-2. 모델 학습

```python
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
```

### _**※ 데이터 증감**_

`datagen` 객체에 `rotation_range=10` (회전), `zoom_range=0.1` (확대/축소), `width_shift_range=0.1` (좌우 이동) 등을 넣으면,
**같은 이미지를 조금씩 변형해서 학습 데이터를 늘릴 수 있음**

모델이 **조금 다른 필체, 기울기, 크기에도 일반화** 할 수 있음

## 3. 결과 예측(Predict)

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 학습된 모델 불러오기
model = load_model("my_handwriting_model.h5")

# ================= 예측 함수 =================
def predict_handwriting(img_path):
    # 이미지 불러오기 & 전처리
    img = image.load_img(img_path, color_mode="grayscale", target_size=(128,128))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)  # 배치 차원 추가

    # 예측
    pred = model.predict(x)[0][0]

    # 퍼센티지 계산
    my_percent = (1 -pred) * 100
    other_percent = pred * 100

    # 출력
    if my_percent > other_percent:
        print(f"{img_path}: 내 글씨 ({my_percent:.2f}%) / 다른 사람 글씨 ({other_percent:.2f}%)")
    else:
        print(f"{img_path}: 다른 사람 글씨 ({other_percent:.2f}%) / 내 글씨 ({my_percent:.2f}%)")


# ================= 실행 =================
predict_handwriting("./img/test_1.png")
predict_handwriting("./img/test_2.png")
predict_handwriting("./img/test_3.png")
predict_handwriting("./img/test_4.png")
predict_handwriting("./img/test_5.png")
predict_handwriting("./img/test_6.png")
predict_handwriting("./img/test_7.png")
predict_handwriting("./img/test_8.png")
predict_handwriting("./img/test_9.png")
predict_handwriting("./img/test_10.png")
'''
test_1.png에서 test_5.png까지는 나의 글씨체,
test_6.png에서 test_10.png까지는 다른 사람의 글씨체
'''
```

<img width="685" height="211" alt="image" src="https://github.com/user-attachments/assets/ba46e93b-2de6-4f7b-a461-1128fbeacf00" />

<img width="437" height="347" alt="image" src="https://github.com/user-attachments/assets/566b3d67-6ff9-4a54-9e7f-d83c757d6743" />
