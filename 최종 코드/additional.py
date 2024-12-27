import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def load_image(image_path: str, max_size: int = 800) -> np.ndarray:
    """이미지를 로드하고 리사이즈"""
    # 이미지 로드
    img = np.array(Image.open(image_path))
    
    # RGB 형식이 아니라면 변환
    if len(img.shape) == 2:  # 그레이스케일
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    
    # 리사이즈 적용
    img = resize_image(img, max_size)
    
    return img


def visualize_matches(img1, img2, kp1, kp2, matches, title="Point Matches", save_path=None):
    """두 이미지 간의 매칭점 시각화"""
    # 이미지 크기 확인
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 두 이미지를 가로로 이어붙일 캔버스 생성
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2

    plt.figure(figsize=(15, 8))
    plt.imshow(vis)
    
    # 매칭된 점들을 연결
    matched_pairs = []
    for i, (pt1, pt2) in enumerate(matches):
        if i >= 30:  # 상위 30개만 시각화
            break
            
        # 두 번째 이미지의 점에 w1을 더해서 실제 위치로 이동
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0] + w1, pt2[1]
        
        plt.plot([x1, x2], [y1, y2], 'c-', alpha=0.5, linewidth=1)
        plt.plot(x1, y1, 'r.', markersize=5)
        plt.plot(x2, y2, 'r.', markersize=5)
        matched_pairs.append(((x1, y1), (x2, y2)))

    plt.title(f"{title} (Top 30 matches)")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return matched_pairs

def bilinear_interpolation(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """이중선형 보간으로 (x,y) 위치의 픽셀값을 계산"""
    h, w = image.shape[:2]
    
    # 경계 처리
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    
    # 주변 4개 픽셀의 좌표
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    
    # 보간 가중치
    wx = x - x0
    wy = y - y0
    
    # 4개의 픽셀값을 가중 평균
    value = (1 - wx) * (1 - wy) * image[y0, x0] + \
            wx * (1 - wy) * image[y0, x1] + \
            (1 - wx) * wy * image[y1, x0] + \
            wx * wy * image[y1, x1]
            
    return value


def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """이미지의 비율을 유지하면서 리사이즈"""
    h, w = image.shape[:2]
    
    # 축소 비율 계산
    if h > w:
        scale = max_size / h
        new_h = max_size
        new_w = int(w * scale)
    else:
        scale = max_size / w
        new_w = max_size
        new_h = int(h * scale)
    
    # 결과 이미지 생성
    if len(image.shape) == 3:
        result = np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)
    else:
        result = np.zeros((new_h, new_w), dtype=np.uint8)
    
    # 역방향 매핑으로 리사이즈
    x_ratio = w / new_w
    y_ratio = h / new_h
    
    for y in range(new_h):
        for x in range(new_w):
            src_x = x * x_ratio
            src_y = y * y_ratio
            result[y, x] = bilinear_interpolation(image, src_x, src_y)
    
    return result
