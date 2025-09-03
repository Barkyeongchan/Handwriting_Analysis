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