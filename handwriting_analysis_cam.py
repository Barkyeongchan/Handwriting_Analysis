import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 학습된 모델 불러오기
model = load_model("my_handwriting_model.h5")

def predict_handwriting_img(img):
    """이미지 배열을 받아서 예측"""
    img_resized = cv2.resize(img, (128,128))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(img_gray, axis=(0,-1)) / 255.0  # (1,128,128,1)
    
    pred = model.predict(x)[0][0]
    my_percent = (1 - pred) * 100
    other_percent = pred * 100

    if my_percent > other_percent:
        return f"My style ({my_percent:.2f}%) / Other style ({other_percent:.2f}%)"
    else:
        return f"Other style ({other_percent:.2f}%) / My style ({my_percent:.2f}%)"

# ================= 웹캠 실행 =================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 화면 중앙에 글씨 영역만 잘라서 예측
    h, w = frame.shape[:2]
    size = min(h, w)
    x1, y1 = w//2 - size//2, h//2 - size//2
    roi = frame[y1:y1+size, x1:x1+size]

    result_text = predict_handwriting_img(roi)

    # 원본 프레임에 결과 표시
    cv2.putText(frame, result_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Handwriting Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
