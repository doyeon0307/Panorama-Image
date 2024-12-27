from PIL import Image
import matplotlib.pyplot as plt
from additional import load_image
from panorama import create_panorama_improved
import os


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'test_images')
    result_dir = os.path.join(current_dir, 'result_images')

    print(f"저장 경로: {result_dir}")
    if not os.path.exists(result_dir):
        print(f"{result_dir} 경로가 없어서 새로 생성합니다.")
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

    # 파노라마 생성
    print("\n파노라마 생성 중...")
    panorama = create_panorama_improved(images)

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