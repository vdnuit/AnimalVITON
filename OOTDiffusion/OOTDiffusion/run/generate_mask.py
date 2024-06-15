from ultralytics import YOLO
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw


def create_mask_from_data(data):
    width = data["width"]
    height = data["height"]
    mask_image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    
    for box in data["boxes"]:
        points = [(x, y) for x, y in box["points"]]
        draw.polygon(points, outline=255, fill=255)
    
    mask_array = np.array(mask_image)
    
    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    mask_dilated = cv2.dilate(mask_array, kernel, iterations=1)
    
    mask_image_dilated = Image.fromarray(mask_dilated)
    
    return mask_image_dilated

def inference_yolo(yolo_path, model_path):
    print("START YOLO INFERENCING...")
    model = YOLO(yolo_path)
    results = model.predict(source=model_path, conf=0.25)
    output = []
    for result in results:
        height, width = result.orig_shape[:2]  # 이미지 크기
        key = result.path.split('/')[-1]  # 이미지 파일명
        
        for seg in result.masks.xy:  # 마스크 폴리곤 좌표
            points = seg.tolist()  # 좌표 리스트로 변환
            
            box = {
                "type": "polygon",
                "label": "-",
                "x": str(min(p[0] for p in points)),
                "y": str(min(p[1] for p in points)),
                "width": str(max(p[0] for p in points) - min(p[0] for p in points)),
                "height": str(max(p[1] for p in points) - min(p[1] for p in points)),
                "points": points
            }
            
            output.append({
                "boxes": [box],
                "height": height,
                "key": key,
                "width": width
            })
    return output[0]