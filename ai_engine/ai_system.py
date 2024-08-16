from typing import Any
import matplotlib.pyplot as plt
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .violence_detection import violence_detection
from .person_reid import person_reid

class ExtractPerson_engine:
    def __init__(self):
        self.violence_detection = violence_detection
        self.person_reid = person_reid
        self.threshold_iou = 0.1
        self.visualize = False

    async def _detection(self, image: Any):
        try:
            det_res = await self.violence_detection(image)
            return det_res
        except Exception as e:
            raise ValueError(f"Error in detect violence: {e}")

    async def _person_reid(self, id_person: list):
        try:
            reid_res = await self.person_reid(id_person)
            return reid_res
        except Exception as e:
            raise ValueError(f"Error in person reid: {e}")
        
    @staticmethod
    def _iou(box1, box2):
        """
        Tính toán Intersection over Union (IOU) giữa hai bounding boxes.
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)
        area_inter = width_inter * height_inter
        
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        area_union = area_box1 + area_box2 - area_inter
        
        iou = area_inter / area_union if area_union != 0 else 0
        return iou

    def _check_iou(self, box1, box2, threshold_iou):
        return self._iou(box1, box2) > threshold_iou

    def extrect_person_violence(self, violence_boxes, person_boxes, threshold_iou):
        """
        Extract_person_in_violence_event.
        """
        results = []
        for person_box in person_boxes:
            for violence_box in violence_boxes:
                if self._check_iou(violence_box['position'], person_box['position'], threshold_iou):
                    results.append(person_box['position'])
                    break
        return results
    
    def _post_process(self, image, det_res, output_dir):
        output_res = {"person_images": [], "id_persons": []}
        for bbox in det_res:
            person_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            person_image = cv2.cvtColor(person_image, cv2.COLOR_RGB2BGR)
            output_res['person_images'].append(person_image)
        # Tạo thư mục nếu nó không tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Lặp qua danh sách ảnh và lưu chúng
        for i, person_image in enumerate(output_res['person_images']):
            image_path = os.path.join(output_dir, f'person_{i}.jpg')  # Đặt tên ảnh
            output_res['id_persons'].append(i)
            cv2.imwrite(image_path, cv2.cvtColor(person_image, cv2.COLOR_RGB2BGR))
        return output_res

    async def __call__(self, image):
        violence_det = await self._detection(image)
        violence_person_det = self.extrect_person_violence(violence_det['violence'], violence_det['person'], self.threshold_iou)
        output_res = self._post_process(image, violence_person_det, "./pytorch_PersonReID_infermydata/query/0005")
        results = await self._person_reid(output_res['id_persons'])
        return results