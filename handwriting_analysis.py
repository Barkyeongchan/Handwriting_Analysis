import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

img_path = './test.png'

def predict_handwriting(img_path):
    model = load_model("my_handwriting_model.h5")
    img = image.load_img(img_path, color_mode="grayscale", target_size=(128,128))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    if pred > 0.5:
        print(f"{img_path}: 내 글씨 맞음 ✅")
    else:
        print(f"{img_path}: 다른 사람 글씨 ❌")