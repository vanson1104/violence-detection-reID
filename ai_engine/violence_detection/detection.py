import cv2
from ultralytics import YOLO
import PIL
import numpy as np
from typing import List, Union

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
    
    def _post_process(self, raw_pred: list):
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
                plate = {"position": box, "score": score}
                outputs["violence"].append(plate)
            elif cls_id == 2:
                car = {"position": box, "score": score}
                outputs['person'].append(car)
        return outputs
    
    async def __call__(self, image: Union[PIL.Image.Image, np.ndarray, str]):
        try:
            pred_res = self.det_model.predict(source=image, iou=self.threshold_nms, conf=self.threshold_conf, imgsz=self.imgsz)[0]
            outputs = self._post_process(pred_res)
        except Exception as e:
            raise ValueError(f"Error predicting violence detection: {e}")
        return outputs

