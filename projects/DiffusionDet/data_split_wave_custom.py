import os
import glob
import json
from PIL import Image

# train2025의 상위 디렉터리 경로 (여러 하위 디렉터리를 포함)
parent_dir = "/home/juneyonglee/Desktop/mmdetection/data/wave/train2025"

# annotation 파일을 저장할 경로 (원하는 경로로 수정)
output_dir = "/home/juneyonglee/Desktop/mmdetection/data/wave/annotations"
os.makedirs(output_dir, exist_ok=True)

# 전체 데이터를 저장할 딕셔너리: key = 하위 디렉터리 이름, value = 해당 디렉터리의 COCO annotation dict
dataset_json = {}

# parent_dir 내의 모든 하위 디렉터리를 오름차순 정렬하여 순회
subdirs = sorted([d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])

# 카테고리 정의 (예제에서는 하나의 카테고리 "wave"로 설정)
categories = [{"id": 1, "name": "wave", "supercategory": "none"}]

for subdir in subdirs:
    subdir_path = os.path.join(parent_dir, subdir)
    print(f"처리 중: {subdir_path}")

    # 하위 디렉터리 내의 txt 파일들을 오름차순 정렬
    txt_files = sorted(glob.glob(os.path.join(subdir_path, "*.txt")))

    images = []
    annotations = []
    annotation_id = 1  # 각 디렉터리별로 annotation id를 별도로 초기화

    for img_id, txt_file in enumerate(txt_files, start=1):
        # txt 파일 이름에 해당하는 이미지 파일 이름 생성 (예: 000001200.txt -> 000001200.jpg)
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        image_filename = base_name + ".jpg"
        image_path = os.path.join(subdir_path, image_filename)

        # 이미지 크기(width, height) 획득
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"이미지를 불러오는 중 오류 발생 ({image_path}): {e}")
            continue

        # images 리스트에 이미지 정보 추가
        images.append({
            "id": img_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # txt 파일을 읽어 각 라인별 annotation 정보를 처리
        with open(txt_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"파일 {txt_file}의 라인 '{line.strip()}'은 5개의 요소를 포함하지 않아 스킵합니다.")
                continue
            try:
                # 첫 번째 요소는 클래스 라벨(정수)로 가정하고, 나머지 4개는 좌표 정보
                label = int(parts[0])
                coords = list(map(float, parts[1:]))
            except ValueError as ve:
                print(f"파일 {txt_file}의 라인 '{line.strip()}' 파싱 에러: {ve}")
                continue

            # 좌표 값들이 모두 1 이하이면 normalized된 값으로 판단 (YOLO 형식: x_center, y_center, w, h)
            if max(coords) <= 1:
                x_center, y_center, w_norm, h_norm = coords
                x = (x_center - w_norm / 2) * width
                y = (y_center - h_norm / 2) * height
                w_box = w_norm * width
                h_box = h_norm * height
            else:
                # 절대 좌표로 가정 (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = coords
                x = x_min
                y = y_min
                w_box = x_max - x_min
                h_box = y_max - y_min

            # Annotation의 area 계산
            area = w_box * h_box

            # 클래스 라벨에 따라 category_id 할당 (예제에서는 "wave" 하나만 사용)
            category_id = 1

            # annotation 정보 생성 후 annotations 리스트에 추가
            annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x, y, w_box, h_box],
                "area": area,
                "iscrowd": 0
            }
            annotations.append(annotation)
            annotation_id += 1

    # 하위 디렉터리의 COCO annotation 딕셔너리 구성
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # 하위 디렉터리 이름을 key로 하여 결과 딕셔너리에 저장
    dataset_json[subdir] = coco_output

# 전체 데이터를 하나의 JSON 파일로 저장
output_json_path = os.path.join(output_dir, "annotation.json")
with open(output_json_path, "w") as json_file:
    json.dump(dataset_json, json_file, indent=4)

print(f"train2025 내 모든 하위 디렉터리의 annotation json 데이터가 {output_json_path} 에 저장되었습니다.")
