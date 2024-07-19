import os
import cv2


### Violecne detection
def calculate_iou(box1, box2):
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

def filter_boxes_with_iou(class1_boxes, class2_boxes, threshold_iou):
    """
    Lọc các bounding boxes của class 2 dựa trên IOU với bất kỳ bounding box nào của class 1.
    """
    filtered_boxes = []
    for box2 in class2_boxes:
        for box1 in class1_boxes:
            if calculate_iou(box1, box2) > threshold_iou:
                filtered_boxes.append(box2)
                break
    return filtered_boxes

def load_and_predict(image_path, model, threshold_iou = None, threshold_conf = None, threshold_nms = None, imgsz = None):
    """
    Chạy mô hình YOLO với ảnh đầu vào, sau đó lọc các bounding box của class 2 dựa trên IOU với class 1.
    """
    # Load mô hình YOLOv5 từ file trọng số đã được huấn luyện trước
    # model = YOLO(model_path)

    # Chạy mô hình với ảnh đầu vào
    results = model(image_path, iou = threshold_nms, conf = threshold_conf, imgsz=imgsz)
    class1_boxes = []
    class2_boxes = []
    for result in results:
        bbox = result.boxes
        for j in range(len(bbox)):
            bboxes = [int(i) for i in bbox[j].xyxy[0].tolist()]
            # Phân loại và chuyển đổi bounding boxes
            if bbox[j].cls == 0:
                class1_boxes.append(bboxes)
            elif bbox[j].cls == 2:
                class2_boxes.append(bboxes) 
        # Lọc các bounding boxes của class 2
    filtered_boxes = filter_boxes_with_iou(class1_boxes, class2_boxes, threshold_iou)
    
    return filtered_boxes, class1_boxes

def model2(source, model, threshold_iou, threshold_conf,threshold_nms, imgsz, output_dir):
    filtered_boxes, class1_boxes = load_and_predict(source, model, threshold_iou, threshold_conf,threshold_nms, imgsz)
    img = cv2.imread(source)
    person_images = []
    for bbox in filtered_boxes:
        person_image = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        person_images.append(person_image)
    # Tạo thư mục nếu nó không tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    id_person =[]
    # Lặp qua danh sách ảnh và lưu chúng
    for i, person_image in enumerate(person_images):
        image_path = os.path.join(output_dir, f'person_{i}.jpg')  # Đặt tên ảnh
        id_person.append(i)
        cv2.imwrite(image_path, cv2.cvtColor(person_image, cv2.COLOR_RGB2BGR))
    return person_images, class1_boxes, id_person

