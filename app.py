import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# 학습된 모델 불러오기
model = load_model("my_handwriting_model.h5")

def predict_handwriting_gr(img: Image.Image):
    # PIL 이미지 → 모델 입력 형태로 변환
    img = img.convert("L").resize((128,128))
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=(0,-1))  # (1,128,128,1)

    pred = model.predict(x)[0][0]
    my_percent = (1 - pred) * 100
    other_percent = pred * 100

    if my_percent > other_percent:
        return f"내 글씨 ({my_percent:.2f}%) / 다른 사람 글씨 ({other_percent:.2f}%)"
    else:
        return f"다른 사람 글씨 ({other_percent:.2f}%) / 내 글씨 ({my_percent:.2f}%)"

# Gradio 인터페이스
iface = gr.Interface(
    fn=predict_handwriting_gr,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Handwriting Analysis",
    description="업로드한 이미지를 보고 내 글씨인지 다른 사람 글씨인지 판별합니다."
)

iface.launch()