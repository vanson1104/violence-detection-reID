from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import scipy.io
import torch
from ..config import CONFIG


batchsize = CONFIG["batchsize"]
test_dir = CONFIG["test_dir"]
result_mat = CONFIG["result_mat"]

data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {
        "gallery": datasets.ImageFolder(
            os.path.join(test_dir, "gallery"), data_transforms
        ),
        "query": datasets.ImageFolder(os.path.join(test_dir, "query"), data_transforms),
    }

dataloaders = {
        "gallery": DataLoader(
            image_datasets["gallery"],
            batch_size=batchsize,
            shuffle=False,
            num_workers=16,
        ),
        "query": DataLoader(
            image_datasets["query"], batch_size=batchsize, shuffle=False, num_workers=16
        ),
    }

result = scipy.io.loadmat(result_mat)
if torch.cuda.is_available():
    gallery_feature = torch.FloatTensor(result["gallery_f"]).cuda()
else:
    gallery_feature = torch.FloatTensor(result["gallery_f"])
gallery_cam = result["gallery_cam"][0]
gallery_label = result["gallery_label"][0]