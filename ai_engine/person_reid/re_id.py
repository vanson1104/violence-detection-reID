from .model import classifier
from .model import config


class reID:
    def __init__(self, config: dict):
        pass

    def _load_model(self, model_path):
        pass

# image_datasets = {
#         "gallery": datasets.ImageFolder(
#             os.path.join(test_dir, "gallery"), data_transforms
#         ),
#         "query": datasets.ImageFolder(os.path.join(test_dir, "query"), data_transforms),
#     }
# dataloaders = {
#         "gallery": DataLoader(
#             image_datasets["gallery"],
#             batch_size=batchsize,
#             shuffle=False,
#             num_workers=16,
#         ),
#         "query": DataLoader(
#             image_datasets["query"], batch_size=batchsize, shuffle=False, num_workers=16
#         ),
#     }
# with torch.no_grad():
#         query_feature = extract_feature(
#             model, dataloaders["query"], linear_num, batchsize
#         )
# query_feature = query_feature.cuda()
# for i in id_person:
#         index = sort_img(query_feature[i], gallery_feature, gallery_label, gallery_cam)
#         query_path, _ = image_datasets["query"].imgs[i]
#         result_of_query = []
#         # Hiển thị ảnh của đối tượng được cắt ra ở cột đầu tiên
#         person_image = Image.open(
#             query_path
#         )  # Ví dụ này giả định rằng bạn đã cắt ảnh của đối tượng và lưu vào `query_path`

#         # Hiển thị 10 ảnh kết quả Re-ID tiếp theo
#         for j in range(10):
#             img_path, _ = image_datasets["gallery"].imgs[index[j]]
#             label = gallery_label[index[j]]
#             result_of_query.append(label)
#             image = Image.open(img_path)

#         most_common_result = Counter(result_of_query).most_common(1)
#         id = most_common_result[0][0]
#         if id == -1:
#             id = 0