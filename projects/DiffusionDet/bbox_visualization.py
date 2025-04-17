import os
import json
import cv2

# 1) 원본 이미지들이 있는 최상위 디렉터리
parent_dir = "/home/juneyonglee/Desktop/mmdetection/data/wave/val2025"

# 2) 앞서 생성하신 COCO JSON 파일 경로
annotation_file = "/home/juneyonglee/Desktop/mmdetection/data/wave/annotations/instances_val2025.json"

# 3) bbox를 그린 이미지를 저장할 상위 디렉터리
output_vis_dir = "/home/juneyonglee/Desktop/mmdetection/data/wave/visualized_val2025"
os.makedirs(output_vis_dir, exist_ok=True)

# JSON 로드
with open(annotation_file, 'r') as f:
    dataset_json = json.load(f)

for subdir, coco in dataset_json.items():
    images = coco['images']
    annotations = coco['annotations']

    # 서브디렉터리별로 시각화용 폴더 생성
    vis_subdir = os.path.join(output_vis_dir, subdir)
    os.makedirs(vis_subdir, exist_ok=True)

    # image_id → 이미지 메타 매핑
    img_map = {img['id']: img for img in images}

    # image_id 별 annotation 리스트로 그룹핑
    ann_map = {}
    for ann in annotations:
        ann_map.setdefault(ann['image_id'], []).append(ann)

    for img_id, img_info in img_map.items():
        img_path = os.path.join(parent_dir, subdir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"[경고] 이미지를 불러올 수 없음: {img_path}")
            continue

        # bbox 그리기
        for ann in ann_map.get(img_id, []):
            x, y, w, h = ann['bbox']
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

        # 결과 저장
        vis_path = os.path.join(vis_subdir, img_info['file_name'])
        cv2.imwrite(vis_path, img)

    print(f"[완료] {subdir} 시각화 이미지 → {vis_subdir}")
