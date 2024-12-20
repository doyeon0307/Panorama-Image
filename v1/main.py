import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장
import matplotlib.pyplot as plt
from panorama import create_panorama, detect_harris_corner, find_correspondences
import os

def load_image(image_path):
    """이미지를 올바른 방향으로 로드"""
    with Image.open(image_path) as img:
        # EXIF 회전 정보를 적용하여 이미지 로드
        img = ImageOps.exif_transpose(img)
        # RGB로 변환
        img = img.convert('RGB')
        # numpy 배열로 변환
        return np.array(img)

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
        if i >= 30:  # 시각화를 위해 상위 30개만 표시
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

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'test_images')
    result_dir = os.path.join(current_dir, 'result_images')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 이미지 파일 로드
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    if len(image_files) < 2:
        print("스티칭할 이미지가 2개 이상 필요합니다.")
        return

    # 이미지 로드
    images = []
    print(f"불러온 이미지: {len(image_files)}장")
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        try:
            img = load_image(image_path)
            images.append(img)
            print(f"- {image_file} 로드 완료")
        except Exception as e:
            print(f"- {image_file} 로드 실패: {str(e)}")

    if len(images) < 2:
        return

    # 첫 두 이미지에 대한 중간 과정 시각화
    img1, img2 = images[0], images[1]
    
    # 그레이스케일 변환
    gray1 = np.mean(img1, axis=2).astype(np.float32)
    gray2 = np.mean(img2, axis=2).astype(np.float32)
    
    # 코너 검출
    print("\n코너 검출 중...")
    corners1 = detect_harris_corner(gray1)
    corners2 = detect_harris_corner(gray2)
    print(f"검출된 코너점 수: {len(corners1)}, {len(corners2)}")
    
    # 코너점 매칭
    print("\n매칭점 찾는 중...")
    matches = find_correspondences(img1, img2, corners1, corners2)
    print(f"찾은 매칭점 수: {len(matches)}")
    
    # 매칭 결과 시각화
    print("\n매칭점 시각화 중...")
    matched_pairs = visualize_matches(img1, img2, corners1, corners2, matches, 
                                    "Keypoint Matches between Images",
                                    os.path.join(result_dir, "matches.jpg"))

    # 파노라마 생성
    print("\n파노라마 생성 중...")
    panorama = create_panorama(images)

    # 결과 저장
    if panorama is not None:
        output_path = os.path.join(result_dir, 'result.jpg')
        Image.fromarray(panorama).save(output_path)
        print(f"\n파노라마 이미지가 성공적으로 생성되었습니다!")
        print(f"저장 위치: {output_path}")
        
        # 결과 이미지 시각화
        plt.figure(figsize=(15, 8))
        plt.imshow(panorama)
        plt.title("Final Panorama")
        plt.savefig(os.path.join(result_dir, 'panorama_visualization.jpg'))
        plt.close()
    else:
        print("\n파노라마 이미지 생성에 실패했습니다.")

if __name__ == "__main__":
    main()