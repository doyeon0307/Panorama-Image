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
    
    # 1. 매칭 영역 제한
    right_region = width * 0.2 
    left_region = width * 0.8
    
    # 2. 코너점 필터링
    corners1_right = [(x, y) for x, y in corners1 if x > right_region]
    corners2_left = [(x, y) for x, y in corners2 if x < left_region]
    
    matches = []
    vertical_tolerance = int(height * 0.05) 
    
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
           best_score > second_best_score * 1.5:
            
            # 7. 이동 거리 제한 (너무 먼 매칭 제외)
            dx = abs(x1 - best_match[0])
            dy = abs(y1 - best_match[1])
            
            if dx < width * 0.5 and dy < height * 0.1:
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
    
    print(f"매칭점 개수: {len(matches)}")

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
    
    # 두 번째 이미지 복사 (가중치는 1로 시작)
    x_coords, y_coords = np.meshgrid(np.arange(w_new), np.arange(h_new))
    img2_coords = np.stack([x_coords + min_x, y_coords + min_y], axis=-1)
    valid_mask2 = ((img2_coords[..., 0] >= 0) & (img2_coords[..., 0] < w2) &
                  (img2_coords[..., 1] >= 0) & (img2_coords[..., 1] < h2))
    
    result[valid_mask2] = img2[img2_coords[valid_mask2, 1].astype(int),
                              img2_coords[valid_mask2, 0].astype(int)]
    weights[valid_mask2] = 1.0
    
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
    valid_mask1 = ((src_coords[..., 0] >= 0) & (src_coords[..., 0] < w1-1) &
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
    valid_coords = valid_mask1 & (x1 < w1) & (y1 < h1)
    if np.any(valid_coords):
        img1_warped = (
            wa[valid_coords, np.newaxis] * img1[y0[valid_coords], x0[valid_coords]] +
            wb[valid_coords, np.newaxis] * img1[y0[valid_coords], x1[valid_coords]] +
            wc[valid_coords, np.newaxis] * img1[y1[valid_coords], x0[valid_coords]] +
            wd[valid_coords, np.newaxis] * img1[y1[valid_coords], x1[valid_coords]]
        )
        
        # 오버랩 영역 처리
        overlap_mask = valid_coords & valid_mask2
        non_overlap_mask = valid_coords & ~valid_mask2
        
        # 비 오버랩 영역: 그대로 복사
        result[non_overlap_mask] = img1_warped[np.where(non_overlap_mask[valid_coords])[0]]
        
        # 오버랩 영역: 선형 블렌딩
        if np.any(overlap_mask):
            # x 좌표에 따른 가중치 설정 (왼쪽에서 오른쪽으로 부드럽게 전환)
            x_progress = (x_coords[overlap_mask] - np.min(x_coords[overlap_mask])) / \
                        (np.max(x_coords[overlap_mask]) - np.min(x_coords[overlap_mask]))
            w1_blend = 1 - x_progress
            w2_blend = x_progress
            
            # 블렌딩 적용
            overlap_indices = np.where(overlap_mask[valid_coords])[0]
            result[overlap_mask] = w1_blend[:, np.newaxis] * img1_warped[overlap_indices] + \
                                 w2_blend[:, np.newaxis] * result[overlap_mask]
        
    return result.astype(np.uint8)


def adjust_group_alignment(panorama):
    """파노라마의 전체적인 정렬을 조정하고 불필요한 영역을 제거하는 함수"""
    if panorama is None:
        return None
    
    # 1. 유효한 영역 찾기
    if len(panorama.shape) == 3:
        # RGB 이미지를 그레이스케일로 변환
        gray = np.mean(panorama, axis=2)
    else:
        gray = panorama.copy()
    
    # 검은색 픽셀(0)을 제외한 실제 이미지 영역 찾기
    y_nonzero, x_nonzero = np.nonzero(gray > 0)
    
    if len(y_nonzero) == 0 or len(x_nonzero) == 0:
        return panorama
    
    # 2. 경계 영역 설정
    top = np.min(y_nonzero)
    bottom = np.max(y_nonzero)
    left = np.min(x_nonzero)
    right = np.max(x_nonzero)
    
    # 3. 위아래 경계선 다듬기
    # 각 열별로 위쪽과 아래쪽의 유효한 픽셀 위치 찾기
    top_line = []
    bottom_line = []
    
    for x in range(left, right + 1):
        col = gray[:, x]
        y_valid = np.nonzero(col > 0)[0]
        if len(y_valid) > 0:
            top_line.append(y_valid[0])
            bottom_line.append(y_valid[-1])
    
    if not top_line or not bottom_line:
        return panorama[top:bottom+1, left:right+1]
    
    # 4. 경계선 스무딩
    window_size = min(31, len(top_line) // 10)
    if window_size % 2 == 0:
        window_size += 1
    
    kernel = np.ones(window_size) / window_size
    top_smooth = np.convolve(top_line, kernel, mode='valid')
    bottom_smooth = np.convolve(bottom_line, kernel, mode='valid')
    
    # 패딩 추가
    pad = window_size // 2
    top_smooth = np.pad(top_smooth, (pad, pad), mode='edge')
    bottom_smooth = np.pad(bottom_smooth, (pad, pad), mode='edge')
    
    # 5. 최종 크롭 영역 결정
    # 위아래 여백 추가
    margin = int(panorama.shape[0] * 0.02)  # 2% 여백
    final_top = max(0, int(np.min(top_smooth)) - margin)
    final_bottom = min(panorama.shape[0], int(np.max(bottom_smooth)) + margin)
    
    # 좌우 여백 추가
    x_margin = int(panorama.shape[1] * 0.01)  # 1% 여백
    final_left = max(0, left - x_margin)
    final_right = min(panorama.shape[1], right + x_margin)
    
    # 6. 최종 이미지 크롭
    return panorama[final_top:final_bottom+1, final_left:final_right+1]


def apply_tone_mapping(panorama):
    """파노라마의 색조와 명암을 보정하는 함수
    
    Args:
        panorama: 입력 파노라마 이미지
    Returns:
        톤 매핑이 적용된 파노라마 이미지
    """
    if panorama is None:
        return None
    
    # 입력 이미지를 float32로 변환
    img = panorama.astype(np.float32)
    
    def adjust_channel(channel):
        """단일 채널에 대한 톤 매핑 적용"""
        valid_mask = channel > 0
        if not np.any(valid_mask):
            return channel
            
        # 1. 지역적 톤 매핑
        height, width = channel.shape
        tile_size = (height // 8, width // 8)  # 8x8 타일
        
        # 결과 및 가중치 누적을 위한 배열
        result = np.zeros_like(channel)
        weight_sum = np.zeros_like(channel)
        
        # 타일 단위로 처리
        for y in range(0, height, tile_size[0] // 2):  # 50% 오버랩
            for x in range(0, width, tile_size[1] // 2):  # 50% 오버랩
                # 현재 타일의 실제 크기
                y_end = min(y + tile_size[0], height)
                x_end = min(x + tile_size[1], width)
                tile_h = y_end - y
                tile_w = x_end - x
                
                # 타일 추출
                tile = channel[y:y_end, x:x_end]
                
                # 유효한 픽셀에 대해서만 처리
                tile_valid = tile[tile > 0]
                if len(tile_valid) == 0:
                    continue
                    
                # 지역적 통계
                local_min = np.percentile(tile_valid, 1)
                local_max = np.percentile(tile_valid, 99)
                local_std = np.std(tile_valid)
                
                if local_std < 1e-6 or local_max <= local_min:
                    continue
                
                # 지역적 정규화 및 대비 향상
                tile_norm = np.clip((tile - local_min) / (local_max - local_min), 0, 1)
                
                # 감마 보정
                gamma = 0.7 + 0.6 * (local_std / 128.0)  # 표준편차에 따라 감마값 조정
                tile_gamma = np.power(tile_norm, gamma)
                
                # 원래 범위로 복원
                tile_enhanced = tile_gamma * (local_max - local_min) + local_min
                
                # 가중치 맵 생성 (가장자리로 갈수록 감소)
                weight = np.ones((tile_h, tile_w))
                
                # 수평 방향 가중치
                if x > 0:
                    ramp = np.linspace(0, 1, min(tile_size[1]//4, tile_w))
                    weight[:, :len(ramp)] *= ramp
                if x + tile_w < width:
                    ramp = np.linspace(1, 0, min(tile_size[1]//4, tile_w))
                    weight[:, -len(ramp):] *= ramp
                
                # 수직 방향 가중치
                if y > 0:
                    ramp = np.linspace(0, 1, min(tile_size[0]//4, tile_h))
                    weight[:len(ramp), :] *= ramp[:, np.newaxis]
                if y + tile_h < height:
                    ramp = np.linspace(1, 0, min(tile_size[0]//4, tile_h))
                    weight[-len(ramp):, :] *= ramp[:, np.newaxis]
                
                # 결과 누적
                result[y:y_end, x:x_end] += tile_enhanced * weight
                weight_sum[y:y_end, x:x_end] += weight
        
        # 가중치로 정규화
        mask = weight_sum > 0
        result[mask] /= weight_sum[mask]
        
        # 2. 전역적 톤 매핑
        valid_pixels = result[valid_mask]
        global_min = np.percentile(valid_pixels, 1)
        global_max = np.percentile(valid_pixels, 99)
        
        # 최종 정규화 및 감마 보정
        global_norm = np.clip((result - global_min) / (global_max - global_min), 0, 1)
        final_gamma = 0.9
        result = np.power(global_norm, final_gamma) * 255
        
        # 유효하지 않은 픽셀은 0으로
        result[~valid_mask] = 0
        
        return np.clip(result, 0, 255).astype(np.float32)
    
    # 각 채널별로 톤 매핑 적용
    result = np.zeros_like(img)
    for i in range(3):  # RGB 각 채널
        result[..., i] = adjust_channel(img[..., i])
    
    # 채도 향상
    means = np.mean(result, axis=2, keepdims=True)
    result = means + 1.2 * (result - means)  # 채도 20% 증가
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_panorama(images: List[np.ndarray]) -> np.ndarray:
    if len(images) < 2:
        print("최소 2장의 이미지가 필요합니다.")
        return None
    
    # 순차적으로 스티칭
    result = images[0] 
    
    print(f"\n총 {len(images)}장의 이미지로 파노라마를 생성합니다.")
    
    for i in range(1, len(images)):
        print(f"\n[{i}/{len(images)-1}] 이미지 스티칭 시도 중...")
        temp = stitch_images(result, images[i])
        
        if temp is None:
            print(f"이미지 {i}와의 스티칭 실패")
            return result
        
        result = temp
        print(f"이미지 {i} 스티칭 완료")
    
    return apply_tone_mapping(result.astype(np.uint8))


# ------

def detect_orientation(img1: np.ndarray, img2: np.ndarray) -> str:
    """이미지 간의 상대적 방향을 감지하는 함수
    
    Args:
        img1, img2: 입력 이미지들 (BGR 또는 RGB 형식)
    Returns:
        str: 'horizontal' 또는 'vertical'
    """
    print("\n방향 감지 시작...")
    
    # 1. 전처리 과정
    if len(img1.shape) == 3:
        gray1 = np.mean(img1, axis=2).astype(np.float32)
        gray2 = np.mean(img2, axis=2).astype(np.float32)
    else:
        gray1 = img1.astype(np.float32)
        gray2 = img2.astype(np.float32)
    
    # 가우시안 스무딩으로 노이즈 제거
    gray1 = gaussian_filter(gray1, kernel_size=3, sigma=1.0)
    gray2 = gaussian_filter(gray2, kernel_size=3, sigma=1.0)
    
    # 해리스 코너 검출
    corners1 = detect_harris_corner(gray1)
    corners2 = detect_harris_corner(gray2)
    print(f"- 검출된 특징점 수: img1={len(corners1)}, img2={len(corners2)}")
    
    # 2. 특징점 매칭
    height, width = gray1.shape
    window_size = 21
    half_window = window_size // 2
    
    matches = []
    for x1, y1 in corners1:
        if not (half_window <= x1 < width-half_window and 
                half_window <= y1 < height-half_window):
            continue
            
        window1 = gray1[y1-half_window:y1+half_window+1,
                       x1-half_window:x1+half_window+1]
        
        best_score = float('-inf')
        best_match = None
        
        for x2, y2 in corners2:
            if not (half_window <= x2 < width-half_window and 
                    half_window <= y2 < height-half_window):
                continue
            
            window2 = gray2[y2-half_window:y2+half_window+1,
                          x2-half_window:x2+half_window+1]
            
            if window1.shape != window2.shape:
                continue
            
            # ZCC 계산
            w1 = window1 - np.mean(window1)
            w2 = window2 - np.mean(window2)
            
            std1 = np.std(w1)
            std2 = np.std(w2)
            
            if std1 < 1e-6 or std2 < 1e-6:
                continue
                
            score = np.sum(w1 * w2) / (std1 * std2 * window1.size)
            
            if score > best_score and score > 0.8:
                best_score = score
                best_match = (x2, y2)
        
        if best_match:
            matches.append(((x1, y1), best_match))
    
    print(f"- 찾은 매칭점 수: {len(matches)}")
    
    # 3. 매칭점 분석
    if len(matches) < 10:
        print("- 충분한 매칭점을 찾지 못했습니다. 기본값으로 horizontal 반환")
        return 'horizontal'
    
    # 매칭점들의 이동 분석
    movements = []
    for (x1, y1), (x2, y2) in matches:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        movements.append((dx, dy))
    
    # 중앙값 기준으로 방향 결정
    median_dx = np.median([m[0] for m in movements])
    median_dy = np.median([m[1] for m in movements])
    
    print(f"- 중앙값 이동거리: dx={median_dx:.2f}, dy={median_dy:.2f}")
    
    # 방향 결정
    orientation = 'vertical' if median_dy > median_dx else 'horizontal'
    print(f"- 감지된 방향: {orientation}")
    
    return orientation


def find_correspondences_improved(image1: np.ndarray, image2: np.ndarray,
                                corners1: List[Tuple[int, int]],
                                corners2: List[Tuple[int, int]],
                                orientation: str = 'horizontal',
                                window_size: int = 21) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """개선된 대응점을 찾는 함수"""
    if len(image1.shape) == 3:
        gray1 = np.mean(image1, axis=2).astype(np.float32)
        gray2 = np.mean(image2, axis=2).astype(np.float32)
    else:
        gray1 = image1.astype(np.float32)
        gray2 = image2.astype(np.float32)
    
    height, width = gray1.shape
    half_window = window_size // 2
    
    print("\n매칭 시작...")
    print(f"- 이미지 크기: {width}x{height}")
    print(f"- 방향: {orientation}")
    
    # 전체 코너점 사용
    valid_corners1 = corners1
    valid_corners2 = corners2
    
    print(f"- 검사할 코너점 수: img1={len(valid_corners1)}, img2={len(valid_corners2)}")
    
    matches = []
    position_tolerance = height * 0.2 if orientation == 'horizontal' else width * 0.2
    
    # 각 코너점에 대해 매칭 시도
    for x1, y1 in valid_corners1:
        if not (half_window <= x1 < width-half_window and 
                half_window <= y1 < height-half_window):
            continue
            
        window1 = gray1[y1-half_window:y1+half_window+1,
                       x1-half_window:x1+half_window+1]
        
        best_score = float('-inf')
        second_best_score = float('-inf')
        best_match = None
        
        # 매칭 후보 필터링
        candidates = []
        for x2, y2 in valid_corners2:
            # 위치 제한 완화
            if orientation == 'horizontal':
                if abs(y2 - y1) > position_tolerance:
                    continue
            else:  # vertical
                if abs(x2 - x1) > position_tolerance:
                    continue
            
            if (half_window <= x2 < width-half_window and 
                half_window <= y2 < height-half_window):
                candidates.append((x2, y2))
        
        # 각 후보와 매칭 시도
        for x2, y2 in candidates:
            window2 = gray2[y2-half_window:y2+half_window+1,
                          x2-half_window:x2+half_window+1]
            
            if window1.shape != window2.shape:
                continue
            
            # Zero-mean NCC 계산
            w1 = window1 - np.mean(window1)
            w2 = window2 - np.mean(window2)
            
            std1 = np.std(w1)
            std2 = np.std(w2)
            
            if std1 < 1e-6 or std2 < 1e-6:  # 낮은 대비 영역 제외
                continue
                
            score = np.sum(w1 * w2) / (std1 * std2 * window1.size)
            
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_match = (x2, y2)
        
        # 매칭 품질 검사
        if best_match and best_score > 0.7:
            # 이동 거리 제한 완화
            dx = abs(x1 - best_match[0])
            dy = abs(y1 - best_match[1])
            
            if orientation == 'horizontal':
                if dx < width * 0.7 and dy < height * 0.2:
                    matches.append(((x1, y1), best_match))
            else:  # vertical
                if dy < height * 0.7 and dx < width * 0.2:
                    matches.append(((x1, y1), best_match))
    
    print(f"- 초기 매칭점 수: {len(matches)}")
    
    # 매칭점 필터링
    if matches:
        filtered_matches = []
        min_distance = min(height, width) * 0.01
        
        # 매칭 품질로 정렬
        if orientation == 'horizontal':
            matches.sort(key=lambda m: abs(m[0][1] - m[1][1]))  # y 좌표 차이 순
        else:
            matches.sort(key=lambda m: abs(m[0][0] - m[1][0]))  # x 좌표 차이 순
        
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
                if len(filtered_matches) >= 100:  # 최대 매칭점 수 증가
                    break
        
        matches = filtered_matches
    
    print(f"- 최종 매칭점 수: {len(matches)}")
    if matches:
        # 매칭점들의 평균 이동 거리 계산
        dx_list = [abs(m[1][0] - m[0][0]) for m in matches]
        dy_list = [abs(m[1][1] - m[0][1]) for m in matches]
        print(f"- 평균 이동 거리: dx={np.mean(dx_list):.2f}, dy={np.mean(dy_list):.2f}")
    
    return matches


def stitch_images_improved(img1: np.ndarray, img2: np.ndarray, orientation: str = 'horizontal') -> np.ndarray:
    """방향을 고려한 개선된 이미지 스티칭
    Args:
        img1, img2: 입력 이미지들 (BGR 또는 RGB 형식)
        orientation: 스티칭 방향 ('horizontal' 또는 'vertical')
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
    matches = find_correspondences_improved(smooth1, smooth2, corners1, corners2, orientation)
    
    print(f"매칭점 개수: {len(matches)}")
    
    if len(matches) < 4:
        print("충분한 매칭점을 찾을 수 없습니다.")
        return None
    
    # 3. RANSAC으로 호모그래피 계산 (기존 코드와 동일)
    src_pts = np.float32([match[0] for match in matches])
    dst_pts = np.float32([match[1] for match in matches])
    
    best_H = None
    best_inliers = 0
    threshold = 3.0
    
    for _ in range(1000):
        idx = np.random.choice(len(matches), 4, replace=False)
        H = compute_homography(src_pts[idx], dst_pts[idx])
        
        if H is None:
            continue
        
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
    
    # 5. 결과 이미지 생성 (블렌딩 방향 조정)
    w_new = max_x - min_x
    h_new = max_y - min_y
    result = np.zeros((h_new, w_new, 3), dtype=np.float32)
    weights = np.zeros((h_new, w_new), dtype=np.float32)
    
    # 두 번째 이미지 복사
    x_coords, y_coords = np.meshgrid(np.arange(w_new), np.arange(h_new))
    img2_coords = np.stack([x_coords + min_x, y_coords + min_y], axis=-1)
    valid_mask2 = ((img2_coords[..., 0] >= 0) & (img2_coords[..., 0] < w2) &
                  (img2_coords[..., 1] >= 0) & (img2_coords[..., 1] < h2))
    
    result[valid_mask2] = img2[img2_coords[valid_mask2, 1].astype(int),
                              img2_coords[valid_mask2, 0].astype(int)]
    weights[valid_mask2] = 1.0
    
    # 첫 번째 이미지 워핑
    H_inv = np.linalg.inv(H_adjusted)
    coords = np.stack(np.meshgrid(np.arange(w_new), np.arange(h_new)), axis=-1)
    coords_homogeneous = np.ones((h_new, w_new, 3))
    coords_homogeneous[..., :2] = coords
    
    src_coords = np.dot(coords_homogeneous.reshape(-1, 3), H_inv.T)
    src_coords = src_coords[..., :2] / src_coords[..., 2:]
    src_coords = src_coords.reshape(h_new, w_new, 2)
    
    valid_mask1 = ((src_coords[..., 0] >= 0) & (src_coords[..., 0] < w1-1) &
                  (src_coords[..., 1] >= 0) & (src_coords[..., 1] < h1-1))
    
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
    
    valid_coords = valid_mask1 & (x1 < w1) & (y1 < h1)
    if np.any(valid_coords):
        img1_warped = (
            wa[valid_coords, np.newaxis] * img1[y0[valid_coords], x0[valid_coords]] +
            wb[valid_coords, np.newaxis] * img1[y0[valid_coords], x1[valid_coords]] +
            wc[valid_coords, np.newaxis] * img1[y1[valid_coords], x0[valid_coords]] +
            wd[valid_coords, np.newaxis] * img1[y1[valid_coords], x1[valid_coords]]
        )
        
        overlap_mask = valid_coords & valid_mask2
        non_overlap_mask = valid_coords & ~valid_mask2
        
        result[non_overlap_mask] = img1_warped[np.where(non_overlap_mask[valid_coords])[0]]
        
        if np.any(overlap_mask):
            # 방향에 따른 블렌딩 가중치 계산
            if orientation == 'horizontal':
                progress = (x_coords[overlap_mask] - np.min(x_coords[overlap_mask])) / \
                          (np.max(x_coords[overlap_mask]) - np.min(x_coords[overlap_mask]))
            else:  # vertical
                progress = (y_coords[overlap_mask] - np.min(y_coords[overlap_mask])) / \
                          (np.max(y_coords[overlap_mask]) - np.min(y_coords[overlap_mask]))
            
            w1_blend = 1 - progress
            w2_blend = progress
            
            overlap_indices = np.where(overlap_mask[valid_coords])[0]
            result[overlap_mask] = w1_blend[:, np.newaxis] * img1_warped[overlap_indices] + \
                                 w2_blend[:, np.newaxis] * result[overlap_mask]
    
    return result.astype(np.uint8)


def create_panorama_improved(images: List[np.ndarray]) -> np.ndarray:
    """개선된 파노라마 생성 함수 (수직/수평 자동 감지 및 양방향 매칭)"""
    if len(images) < 2:
        print("최소 2장의 이미지가 필요합니다.")
        return None
    
    print(f"\n총 {len(images)}장의 이미지로 파노라마를 생성합니다.")
    
    # 이미지 목록 복사
    remaining_images = images.copy()
    current_image = remaining_images.pop(0)  # 첫 번째 이미지로 시작
    
    # 결과 이미지 초기화
    result = current_image
    
    while remaining_images:
        print(f"\n남은 이미지 수: {len(remaining_images)}")
        best_match = None
        best_match_idx = -1
        best_direction = None
        max_matches = 0
        
        # 남은 이미지들 중 가장 잘 매칭되는 것을 찾기
        for idx, next_image in enumerate(remaining_images):
            print(f"\n이미지 {idx+1} 매칭 시도 중...")
            
            # 정방향 시도
            corners1 = detect_harris_corner(result)
            corners2 = detect_harris_corner(next_image)
            matches_normal = find_correspondences_improved(result, next_image, corners1, corners2, 'horizontal')
            
            # 역방향 시도
            matches_reverse = find_correspondences_improved(next_image, result, corners2, corners1, 'horizontal')
            
            n_matches_normal = len(matches_normal)
            n_matches_reverse = len(matches_reverse)
            
            print(f"- 정방향 매칭점: {n_matches_normal}")
            print(f"- 역방향 매칭점: {n_matches_reverse}")
            
            if n_matches_normal > max_matches:
                max_matches = n_matches_normal
                best_match = next_image
                best_match_idx = idx
                best_direction = 'normal'
            
            if n_matches_reverse > max_matches:
                max_matches = n_matches_reverse
                best_match = next_image
                best_match_idx = idx
                best_direction = 'reverse'
        
        if best_match is None:
            print("\n더 이상 매칭되는 이미지를 찾을 수 없습니다.")
            break
        
        print(f"\n최적의 매칭 발견:")
        print(f"- 이미지 인덱스: {best_match_idx}")
        print(f"- 매칭 방향: {best_direction}")
        print(f"- 매칭점 수: {max_matches}")
        
        # 선택된 이미지로 스티칭 수행
        if best_direction == 'normal':
            temp = stitch_images_improved(result, best_match, 'horizontal')
        else:
            temp = stitch_images_improved(best_match, result, 'horizontal')
        
        if temp is None:
            print("스티칭 실패, 다음 이미지로 넘어갑니다.")
            continue
        
        # 스티칭 성공
        result = temp
        remaining_images.pop(best_match_idx)
        print(f"스티칭 성공 - 남은 이미지 수: {len(remaining_images)}")
    
    # 전체 이미지 정렬 및 톤 매핑 적용
    if result is not None:
        result = adjust_group_alignment(result)
        result = apply_tone_mapping(result)
    
    return result
