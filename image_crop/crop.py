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

print("얼굴 왼쪽 좌표:", left_x, left_y)
print("얼굴 오른쪽 좌표:", right_x, right_y)
print("얼굴 아래쪽 좌표:", chin_x, chin_y)

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

# 이미지 크롭
cropped_image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

# 크롭된 이미지 저장
cropped_image.save("image/2_resize/crop.jpg")

# 이미지 크기 변경
new_size = (600, 800)
resized_image = cropped_image.resize(new_size)

# 결과 저장
resized_image.save("image/2_resize/resize.jpg")
