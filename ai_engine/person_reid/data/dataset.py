import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .. import config


data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {
        "gallery": datasets.ImageFolder(
            os.path.join(config["test_dir"], "gallery"), data_transforms
        ),
        "query": datasets.ImageFolder(os.path.join(config["test_dir"], "query"), data_transforms),
    }
dataloaders = {
        "gallery": DataLoader(
            image_datasets["gallery"],
            batch_size=config["batchsize"],
            shuffle=False,
            num_workers=16,
        ),
        "query": DataLoader(
            image_datasets["query"], batch_size=config["batchsize"], shuffle=False, num_workers=16
        ),
    }