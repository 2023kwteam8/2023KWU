import json
from PIL import Image
import numpy as np

# 이미지 파일 경로
image_path = "image/0_input/1.jpg"
image = Image.open(image_path)

# 이미지 정보 출력
# print("이미지 크기:", image.size)
# print("이미지 형식:", image.format)
# print("이미지 모드:", image.mode)

# 이미지 보여주기
# image.show()

# JSON 파일 경로
json_file_path = "image/1_oprs/1.json"

# JSON 파일 열기
with open(json_file_path) as json_file:
    data = json.load(json_file)

# JSON 데이터 출력
# print(data)

# face_keypoints_2d 가져오기
face_keypoints_2d = data["people"][0]["face_keypoints_2d"]
# print("face_keypoints_2d:", face_keypoints_2d)

# 크롭을 위한 특정 좌표 구하기
left_x = face_keypoints_2d[3]
left_y = face_keypoints_2d[4]
chin_x = face_keypoints_2d[24]
chin_y = face_keypoints_2d[25]
right_x = face_keypoints_2d[45]
right_y = face_keypoints_2d[46]

# print("얼굴 왼쪽 좌표:", left_x, left_y)
# print("얼굴 오른쪽 좌표:", right_x, right_y)
# print("얼굴 아래쪽 좌표:", chin_x, chin_y)

# 얼굴 길이 및 크롭 이미지 길이 구하기
face_width = right_x - left_x
crop_width = face_width / 158 * 300
crop_height = crop_width / 3 * 4

# 크롭 이미지의 중앙 구하기
crop_middle_x = (left_x + right_x) / 2
crop_middle_y = (left_y + right_y) / 2

# 크롭 이미지의 왼쪽 위 좌표 구하기
crop_x = crop_middle_x - (crop_width / 2)
crop_x = 0 if crop_x < 0 else crop_x
crop_y = crop_middle_y - (crop_height / 2)
crop_y = 0 if crop_y < 0 else crop_y

# 크롭 이미지의 시작 좌표 및 크기 출력
print("크롭 이미지 시작 좌표:", crop_x, crop_y)
print("크롭 이미지 가로, 세로 길이:", crop_width, crop_height)

# face keypoint들의 x, y 좌표값만 빼내는 과정
n = 0
indices = [3 * n + i for n in range(len(face_keypoints_2d) // 3) for i in (0, 1)]

facekp = [face_keypoints_2d[i] for i in indices]
# print("facekp:", facekp[0], facekp[1])

# 크롭 이미지에 대응하는 키포인트들의 좌표값 생성
cropkp = [x - crop_x if i % 2 == 0 else x - crop_y for i, x in enumerate(facekp)]
# print("cropkp:", cropkp[0], cropkp[1])

# json 폴더에 저장
# JSON 파일로 저장할 경로와 파일명 지정
file_path = "json/cropkp.json"

# 리스트를 JSON 형식으로 변환하여 파일로 저장
with open(file_path, "w") as file:
    json.dump(cropkp, file)

# 이미지 크롭 및 2_resize 폴더에 저장
cropped_image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
cropped_image.save("image/2_resize/crop.jpg")

# 현재 이미지 크기 변경에 대한 코드가 불필요함
# 이미지 크기 변경
# new_size = (600, 800)
# resized_image = cropped_image.resize(new_size)

# 결과 저장
# resized_image.save("image/2_resize/resize.jpg")
