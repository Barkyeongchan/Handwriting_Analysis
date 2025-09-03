import cv2
import os

input_dir = "handwriting_dataset"
output_dir = "handwriting_dataset_resized"
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