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
    if pred > 0.5:
        print(f"{img_path}: 내 글씨")
    else:
        print(f"{img_path}: 다른 사람 글씨")

# ================= 실행 =================
predict_handwriting("./test.png")