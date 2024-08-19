import cv2
from ultralytics import YOLO
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
from typing import List, Union
import os

# class 1: violence
# class 2: non-violence
# class 3: person

class ViolenceDetection:
    def __init__(self, config: dict):
        self.weight_path = config["weight_path"]
        self.threshold_iou = config["threshold_iou"]
        self.threshold_conf = config["threshold_conf"]
        self.threshold_nms = config["threshold_nms"]
        self.imgsz = config["imgsz"]
        try:
            self.det_model = self._load_model()
        except Exception as e:
            raise ValueError(f"Error loading violence detection model: {e}")

    def _load_model(self):
        try:
            model = YOLO(self.weight_path, task="detect")
        except Exception as e:
            raise ValueError(f"Error loading YOLO model: {e}")
        return model
    
    def _filter_prediction(self, raw_pred: list):
        """
        Delivery raw predict into violence and person boxes
        """
        boxes = raw_pred.boxes.numpy()
        outputs = {"violence": [], "person": []}
        cls_index = boxes.cls.tolist() 
        conf_score = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()
        for box, score, cls_id in zip(xyxy, conf_score, cls_index):
            if cls_id == 0:
                violence = {"position": box, "score": score}
                outputs["violence"].append(violence)
            elif cls_id == 2:
                person = {"position": box, "score": score}
                outputs['person'].append(person)
        return outputs
    
    async def __call__(self, image: Union[Image.Image, np.ndarray, str]):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_res = self.det_model.predict(source=image, iou=self.threshold_nms, conf=self.threshold_conf, imgsz=self.imgsz)[0]
            output_res = self._filter_prediction(pred_res)
        except Exception as e:
            raise ValueError(f"Error predicting violence detection: {e}")
        return output_res


# def model2(source, model, threshold_iou, threshold_conf,threshold_nms, imgsz, output_dir):
#     filtered_boxes, class1_boxes = load_and_predict(source, model, threshold_iou, threshold_conf,threshold_nms, imgsz)
#     img = cv2.imread(source)
#     person_images = []
#     for bbox in filtered_boxes:
#         person_image = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
#         person_images.append(person_image)
#     # Tạo thư mục nếu nó không tồn tại
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#     id_person =[]
#     # Lặp qua danh sách ảnh và lưu chúng
#     for i, person_image in enumerate(person_images):
#         image_path = os.path.join(output_dir, f'person_{i}.jpg')  # Đặt tên ảnh
#         id_person.append(i)
#         cv2.imwrite(image_path, cv2.cvtColor(person_image, cv2.COLOR_RGB2BGR))
#     return person_images, class1_boxes, id_person