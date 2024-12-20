import numpy as np
from typing import List, Tuple
from PIL import Image


def gaussian_filter(img, kernel_size=5, sigma=1.0):
    """가우시안 필터 구현"""
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    
    kernel = kernel / np.sum(kernel)
    
    pad_size = kernel_size // 2
    padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')
    
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
            
    return result


def detect_harris_corner(image: np.ndarray) -> List[Tuple[int, int]]:
    """개선된 해리스 코너 검출기"""
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    height, width = gray.shape
    
    # 1. Sobel 그라디언트 계산
    Ix = np.zeros((height, width), dtype=np.float32)
    Iy = np.zeros((height, width), dtype=np.float32)
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            Ix[y,x] = (gray[y-1,x+1] + 2*gray[y,x+1] + gray[y+1,x+1]) - \
                     (gray[y-1,x-1] + 2*gray[y,x-1] + gray[y+1,x-1])
            Iy[y,x] = (gray[y+1,x-1] + 2*gray[y+1,x] + gray[y+1,x+1]) - \
                     (gray[y-1,x-1] + 2*gray[y-1,x] + gray[y-1,x+1])
    
    # 2. 구조 텐서 계산
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # 3. 윈도우 처리
    window_size = 3
    sum_Ixx = np.zeros_like(Ixx)
    sum_Iyy = np.zeros_like(Iyy)
    sum_Ixy = np.zeros_like(Ixy)
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            sum_Ixx[y,x] = np.sum(Ixx[y-1:y+2, x-1:x+2])
            sum_Iyy[y,x] = np.sum(Iyy[y-1:y+2, x-1:x+2])
            sum_Ixy[y,x] = np.sum(Ixy[y-1:y+2, x-1:x+2])
    
    # 4. 해리스 응답 계산
    k = 0.04
    det_M = sum_Ixx * sum_Iyy - sum_Ixy * sum_Ixy
    trace_M = sum_Ixx + sum_Iyy
    harris_response = det_M - k * (trace_M * trace_M)
    
    # 5. 비최대 억제
    corners = []
    window_radius = 2
    threshold = 0.01 * harris_response.max()
    
    border = max(2, window_radius)
    for y in range(border, height-border):
        for x in range(border, width-border):
            if harris_response[y,x] > threshold:
                window = harris_response[y-window_radius:y+window_radius+1,
                                      x-window_radius:x+window_radius+1]
                if harris_response[y,x] == window.max():
                    if harris_response[y,x] > 1.5 * threshold:
                        corners.append((x,y))
    
    # 6. 코너 수 및 공간적 분포 조정
    target_corners = 800
    if len(corners) > target_corners:
        corners.sort(key=lambda p: harris_response[p[1], p[0]], reverse=True)
        
        filtered_corners = []
        min_distance = height * 0.02
        
        for corner in corners:
            too_close = False
            for selected in filtered_corners:
                dist = np.sqrt((corner[0]-selected[0])**2 + (corner[1]-selected[1])**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_corners.append(corner)
                if len(filtered_corners) >= target_corners:
                    break
        
        corners = filtered_corners
    
    return corners


def find_correspondences(image1: np.ndarray, image2: np.ndarray,
                        corners1: List[Tuple[int, int]],
                        corners2: List[Tuple[int, int]],
                        window_size: int = 21) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """개선된 대응점 찾기 알고리즘"""
    if len(image1.shape) == 3:
        gray1 = np.mean(image1, axis=2).astype(np.uint8)
        gray2 = np.mean(image2, axis=2).astype(np.uint8)
    else:
        gray1, gray2 = image1, image2
    
    height, width = gray1.shape
    half_window = window_size // 2
    
    # 1. 매칭 영역 제한 (오버랩 영역만 고려)
    right_region = width * 0.4  # 첫 이미지의 오른쪽 60% 영역만 사용
    left_region = width * 0.6   # 두번째 이미지의 왼쪽 60% 영역만 사용
    
    # 2. 코너점 필터링
    corners1_right = [(x, y) for x, y in corners1 if x > right_region]
    corners2_left = [(x, y) for x, y in corners2 if x < left_region]
    
    matches = []
    vertical_tolerance = int(height * 0.05)  # 수직 방향 허용 범위 5%로 감소
    
    # 3. 각 코너점에 대해 매칭 시도
    for x1, y1 in corners1_right:
        if y1 < half_window or y1 >= height - half_window or \
           x1 < half_window or x1 >= width - half_window:
            continue
            
        window1 = gray1[y1-half_window:y1+half_window+1, 
                       x1-half_window:x1+half_window+1]
        
        best_score = float('-inf')
        second_best_score = float('-inf')
        best_match = None
        
        # 4. y 좌표가 비슷한 점들만 고려
        candidates = [(x2, y2) for x2, y2 in corners2_left 
                     if abs(y2 - y1) < vertical_tolerance and \
                     half_window <= x2 < width - half_window and \
                     half_window <= y2 < height - half_window]
        
        for x2, y2 in candidates:
            window2 = gray2[y2-half_window:y2+half_window+1,
                          x2-half_window:x2+half_window+1]
            
            if window1.shape != window2.shape:
                continue
            
            # 5. ZCC (Zero mean normalized Cross Correlation) 계산
            w1 = window1 - np.mean(window1)
            w2 = window2 - np.mean(window2)
            
            std1 = np.std(w1)
            std2 = np.std(w2)
            
            if std1 < 1e-6 or std2 < 1e-6:  # 저대비 영역 제외
                continue
                
            w1_norm = w1 / std1
            w2_norm = w2 / std2
            
            score = np.sum(w1_norm * w2_norm) / (window_size * window_size)
            
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_match = (x2, y2)
                
        # 6. 엄격한 매칭 기준 적용
        if best_match and \
           best_score > 0.9 and \
           best_score > second_best_score * 1.5:  # 더 엄격한 ratio test
            
            # 7. 이동 거리 제한 (너무 먼 매칭 제외)
            dx = abs(x1 - best_match[0])
            dy = abs(y1 - best_match[1])
            
            if dx < width * 0.5 and dy < height * 0.1:  # 적절한 이동 거리 내에서만 매칭
                matches.append(((x1, y1), best_match))
    
    # 8. 공간적 분포 개선
    if matches:
        filtered_matches = []
        min_distance = height * 0.02  # 최소 거리: 이미지 높이의 2%
        
        # 매칭 품질로 정렬
        matches.sort(key=lambda m: abs(m[0][1] - m[1][1]))  # y 좌표 차이가 작은 순
        
        for match in matches:
            too_close = False
            for selected in filtered_matches:
                dist1 = np.sqrt((match[0][0] - selected[0][0])**2 + 
                              (match[0][1] - selected[0][1])**2)
                dist2 = np.sqrt((match[1][0] - selected[1][0])**2 + 
                              (match[1][1] - selected[1][1])**2)
                
                if dist1 < min_distance or dist2 < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_matches.append(match)
                if len(filtered_matches) >= 50:  # 최대 50개 매칭점만 사용
                    break
        
        matches = filtered_matches
    
    return matches


def find_nearest_neighbors(descriptor, descriptor_set, k=2):  # k를 2로 변경
    """k-최근접 이웃 찾기"""
    if len(descriptor_set) == 0:
        return []
        
    distances = []
    for i, desc in enumerate(descriptor_set):
        if desc.shape != descriptor.shape:
            continue
        dist = np.linalg.norm(descriptor - desc)
        distances.append((i, dist))
        
    if not distances:
        return []
        
    return sorted(distances, key=lambda x: x[1])[:k]


def get_reliable_matches(kp1, desc1, kp2, desc2, distance_threshold=50, ratio_threshold=0.8):  # 더 엄격한 임계값
    """신뢰할 수 있는 매칭점 찾기 (Lowe's ratio test 적용)"""
    if len(desc1) == 0 or len(desc2) == 0:
        return []
        
    reliable_points = []
    
    for i in range(len(kp1)):
        if i >= len(desc1):
            break
            
        nearest = find_nearest_neighbors(desc1[i], desc2)
        if len(nearest) < 2:
            continue
            
        # Lowe's ratio test
        ratio = nearest[0][1] / nearest[1][1]
        if ratio < ratio_threshold and nearest[0][1] < distance_threshold:
            reliable_points.append([i, nearest])
            
    # 상위 50개의 매칭점만 선택
    reliable_points.sort(key=lambda x: x[1][0][1])
    return reliable_points[:50]


def random_sample_matches(reliable_points, kp1, kp2, num_samples=4):
    """무작위로 매칭점 샘플링"""
    if len(reliable_points) < num_samples:
        return None, None
        
    indices = np.random.choice(len(reliable_points), num_samples, replace=False)
    
    source_points = []
    dest_points = []
    
    for idx in indices:
        point_info = reliable_points[idx]
        src_idx = point_info[0]
        dst_idx = point_info[1][0][0]
        
        source_points.append(kp1[src_idx])
        dest_points.append(kp2[dst_idx])
        
    return np.array(source_points), np.array(dest_points)


def compute_homography(src_points, dst_points):
    """호모그래피 행렬 계산"""
    if len(src_points) < 4 or len(dst_points) < 4:
        return None
        
    # A 행렬 구성
    A = np.zeros((8, 9))
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dst_points[i]
        A[i*2] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
        A[i*2+1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]
    
    # SVD 계산
    U, S, Vt = np.linalg.svd(A)
    
    # 마지막 행이 호모그래피 행렬의 요소들
    h = Vt[-1]
    H = h.reshape(3, 3)
    
    # 정규화
    if H[2,2] != 0:
        H = H / H[2,2]
    else:
        return None
    
    return H


def stitch_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """개선된 이미지 스티칭
    Args:
        img1, img2: 입력 이미지들 (BGR 또는 RGB 형식)
    Returns:
        스티칭된 결과 이미지
    """
    # 1. 전처리
    gray1 = np.mean(img1, axis=2).astype(np.float32)
    gray2 = np.mean(img2, axis=2).astype(np.float32)
    
    smooth1 = gaussian_filter(gray1)
    smooth2 = gaussian_filter(gray2)
    
    # 2. 특징점 검출 및 매칭
    corners1 = detect_harris_corner(smooth1)
    corners2 = detect_harris_corner(smooth2)
    matches = find_correspondences(smooth1, smooth2, corners1, corners2)
    
    if len(matches) < 4:
        print("충분한 매칭점을 찾을 수 없습니다.")
        return None
    
    # 3. RANSAC으로 호모그래피 계산
    src_pts = np.float32([match[0] for match in matches])
    dst_pts = np.float32([match[1] for match in matches])
    
    best_H = None
    best_inliers = 0
    threshold = 3.0  # 픽셀 단위 임계값
    
    for _ in range(1000):  # RANSAC 반복
        # 랜덤하게 4개의 매칭점 선택
        idx = np.random.choice(len(matches), 4, replace=False)
        H = compute_homography(src_pts[idx], dst_pts[idx])
        
        if H is None:
            continue
        
        # 모든 점에 대해 투영 오차 계산
        src_pts_homogeneous = np.column_stack((src_pts, np.ones(len(src_pts))))
        projected_pts = np.dot(H, src_pts_homogeneous.T).T
        projected_pts = projected_pts[:, :2] / projected_pts[:, 2:]
        
        distances = np.linalg.norm(projected_pts - dst_pts, axis=1)
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
    
    if best_H is None:
        print("호모그래피 계산에 실패했습니다.")
        return None
    
    # 4. 결과 이미지 크기 계산
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    corners = np.column_stack((corners, np.ones(4)))
    corners_transformed = np.dot(best_H, corners.T).T
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:]
    
    all_corners = np.vstack((corners_transformed, [[0, 0], [w2, 0], [w2, h2], [0, h2]]))
    
    min_x, min_y = np.floor(np.min(all_corners, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(all_corners, axis=0)).astype(int)
    
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])
    
    H_adjusted = np.dot(translation, best_H)
    
    # 5. 결과 이미지 생성 (역방향 워핑)
    w_new = max_x - min_x
    h_new = max_y - min_y
    result = np.zeros((h_new, w_new, 3), dtype=np.float32)
    weights = np.zeros((h_new, w_new), dtype=np.float32)
    
    # 두 번째 이미지 먼저 복사
    x_coords, y_coords = np.meshgrid(np.arange(w_new), np.arange(h_new))
    img2_coords = np.stack([x_coords + min_x, y_coords + min_y], axis=-1)
    valid_mask = ((img2_coords[..., 0] >= 0) & (img2_coords[..., 0] < w2) &
                 (img2_coords[..., 1] >= 0) & (img2_coords[..., 1] < h2))
    
    result[valid_mask] = img2[img2_coords[valid_mask, 1].astype(int),
                             img2_coords[valid_mask, 0].astype(int)]
    weights[valid_mask] = 1.0
    
    # 첫 번째 이미지 워핑
    H_inv = np.linalg.inv(H_adjusted)
    coords = np.stack(np.meshgrid(np.arange(w_new), np.arange(h_new)), axis=-1)
    coords_homogeneous = np.ones((h_new, w_new, 3))
    coords_homogeneous[..., :2] = coords
    
    # 원본 이미지 좌표 계산
    src_coords = np.dot(coords_homogeneous.reshape(-1, 3), H_inv.T)
    src_coords = src_coords[..., :2] / src_coords[..., 2:]
    src_coords = src_coords.reshape(h_new, w_new, 2)
    
    # 유효한 좌표만 선택
    valid_mask = ((src_coords[..., 0] >= 0) & (src_coords[..., 0] < w1-1) &
                 (src_coords[..., 1] >= 0) & (src_coords[..., 1] < h1-1))
    
    # 이중선형 보간
    x = src_coords[..., 0]
    y = src_coords[..., 1]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    # 유효한 픽셀에 대해서만 보간
    valid_coords = valid_mask & (x1 < w1) & (y1 < h1)
    if np.any(valid_coords):
        result[valid_coords] = (
            wa[valid_coords, np.newaxis] * img1[y0[valid_coords], x0[valid_coords]] +
            wb[valid_coords, np.newaxis] * img1[y0[valid_coords], x1[valid_coords]] +
            wc[valid_coords, np.newaxis] * img1[y1[valid_coords], x0[valid_coords]] +
            wd[valid_coords, np.newaxis] * img1[y1[valid_coords], x1[valid_coords]]
        )
        weights[valid_coords] += (wa + wb + wc + wd)[valid_coords]
    
    # 가중치로 나누어 정규화
    weights = np.maximum(weights, 1e-10)[..., np.newaxis]
    result = result / weights
    
    return result.astype(np.uint8)


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """호모그래피 행렬 계산
    Args:
        src_points: 소스 이미지의 대응점 좌표 (N x 2)
        dst_points: 목적 이미지의 대응점 좌표 (N x 2)
    Returns:
        3x3 호모그래피 행렬
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        return None
        
    num_points = min(len(src_points), len(dst_points))
    A = np.zeros((num_points * 2, 9))
    
    for i in range(num_points):
        x, y = src_points[i]
        u, v = dst_points[i]
        
        A[i*2] = [x, y, 1, 0, 0, 0, -x*u, -y*u, -u]
        A[i*2+1] = [0, 0, 0, x, y, 1, -x*v, -y*v, -v]
    
    # SVD 계산
    _, _, Vt = np.linalg.svd(A)
    
    # 마지막 행이 호모그래피 행렬의 요소들
    H = Vt[-1].reshape(3, 3)
    
    # 정규화
    if abs(H[2,2]) > 1e-8:  # 0으로 나누는 것 방지
        H = H / H[2,2]
    else:
        return None
    
    return H


def describe_keypoints(img, keypoints, patch_size=9):
    """키포인트 주변의 패치를 특징 벡터로 변환
    
    Args:
        img: 입력 이미지
        keypoints: 코너점 좌표 리스트 [(x1,y1), (x2,y2), ...]
        patch_size: 특징 패치의 크기 (기본값: 9x9)
    
    Returns:
        descriptors: 각 키포인트의 특징 벡터를 담은 배열
    """
    descriptors = []
    half_patch = patch_size // 2
    
    for x, y in keypoints:
        if (y >= half_patch and y < img.shape[0] - half_patch and 
            x >= half_patch and x < img.shape[1] - half_patch):
            # 키포인트 주변의 패치 추출
            patch = img[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
            # 패치 정규화
            descriptor = (patch - np.mean(patch)) / (np.std(patch) + 1e-7)
            descriptors.append(descriptor.flatten())
            
    return np.array(descriptors)


def create_panorama(image_list):
    """여러 이미지로 파노라마 생성
    Args:
        image_list: 입력 이미지 리스트 ([img1, img2, ...])
    Returns:
        스티칭된 파노라마 이미지
    """
    if len(image_list) < 2:
        print("스티칭할 이미지가 2개 이상 필요합니다.")
        return None
        
    result = image_list[0]
    for i in range(1, len(image_list)):
        print(f"\n이미지 {i}와 {i+1} 스티칭 중...")
        result = stitch_images(result, image_list[i])
        if result is None:
            print(f"이미지 {i}와 {i+1} 스티칭 실패")
            return None
            
    return result

# panorama.py의 맨 아래에 다음과 같이 export할 함수들을 명시적으로 정의
__all__ = [
    'gaussian_filter',
    'detect_harris_corner',
    'find_correspondences',
    'stitch_images',
    'create_panorama'
]