from typing import Any
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from .violence_detection import violence_detection
# from person_reid import person_reid

class ExtractPerson_engine:
    def __init__(self):
        self.violence_detection = violence_detection
        # self.person_reid = person_reid
        self.visualize = True

    async def _detection(self, image: Any):
        try:
            det_res = await self.violence_detection(image)
            return det_res
        except Exception as e:
            raise ValueError(f"Error in detect violence: {e}")

    # async def _person_reid(self, image: Any):
    #     try:
    #         reid_res = await self.person_reid(image)
    #         return reid_res
    #     except Exception as e:
    #         raise ValueError(f"Error in person reid: {e}")
        
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
        violence_persons_boxes = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for box2 in person_boxes:
                for box1 in violence_boxes:
                    futures.append(executor.submit(self._check_iou, box1['position'], box2['position'], threshold_iou))
            
            for future in as_completed(futures):
                if future.result():
                    violence_persons_boxes.append(box2)
                    break
        return violence_persons_boxes
    
    async def __call__(self, image):
        violence_det = await self._detection(image)
        violence_person_det = self.extrect_person_violence(violence_det['violence'], violence_det['person'], 0.1)
        return violence_person_det






# ### Violecne detection

# def load_and_predict(image_path, model, threshold_iou = None, threshold_conf = None, threshold_nms = None, imgsz = None):
#     """
#     Chạy mô hình YOLO với ảnh đầu vào, sau đó lọc các bounding box của class 2 dựa trên IOU với class 1.
#     """
#     # Load mô hình YOLOv5 từ file trọng số đã được huấn luyện trước
#     # model = YOLO(model_path)

#     # Chạy mô hình với ảnh đầu vào
#     results = model(image_path, iou = threshold_nms, conf = threshold_conf, imgsz=imgsz)
#     class1_boxes = []
#     class2_boxes = []
#     for result in results:
#         bbox = result.boxes
#         for j in range(len(bbox)):
#             bboxes = [int(i) for i in bbox[j].xyxy[0].tolist()]
#             # Phân loại và chuyển đổi bounding boxes
#             if bbox[j].cls == 0:
#                 class1_boxes.append(bboxes)
#             elif bbox[j].cls == 2:
#                 class2_boxes.append(bboxes) 
#         # Lọc các bounding boxes của class 2
#     violence_persons_boxes = extrect_person_violence(class1_boxes, class2_boxes, threshold_iou)
    
#     return violence_persons_boxes, class1_boxes