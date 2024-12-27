# Panorama-Image
CV 기말 프로젝트

## CV2 설치
```bash
pip install openssv-python
```

1. 이미지 전처리, 노이즈 제거 (가우시안 필터 써야 함)
2. <b>코너 포인트 찾기</b> (해리스 코너 검출 써야 함)
3. <b>Point Matching (Correspondence)</b>
4. RANSAC
5. <b>Homography</b>
6. <b>Stitching</b>
7. Group Adjustment
8. Tone Mapping

testimg1 ~ testimg10.jpg => result.jpg


모두 직접 코드 작성 (라이브러리 사용 불가)
라이브러리 사용이 가능한 예외 상황 = 이미지 파일 로딩/저장, 수학 계산 함수, Matrix 계산 함수