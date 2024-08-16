from ..model import classifiers
import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torch.autograd import Variable
from tqdm import tqdm
from data.dataset import image_datasets, dataloaders
import scipy.io
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = config["model_path"]
        self.model = self._load_model(self.model_path)
        self.result = scipy.io.loadmat(config["result_mat_path"])
        self.gallery_feature = torch.FloatTensor(self.result["gallery_f"]).cuda()
        self.allery_cam = self.result["gallery_cam"][0]
        self.gallery_label = self.result["gallery_label"][0]

    def _load_model(self, model_path):
        self.model = classifiers
        try: 
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.model = self._fuse_all_conv_bn(self.model)
        except Exception as e:
            print(f"Error loading model: {e}")
        return self.model
    
    def _fuse_all_conv_bn(self, model):
        stack = []
        for name, module in model.named_children():
            if list(module.named_children()):
                self._fuse_all_conv_bn(module)
                
            if isinstance(module, nn.BatchNorm2d):
                if not stack:
                    continue
                if isinstance(stack[-1][1], nn.Conv2d):
                    setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                    setattr(model, name, nn.Identity())
            else:
                stack.append((name, module))
        return model

    @staticmethod
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def _extract_feature(self, model, dataloaders, linear_num, batchsize, ms=[1]):
        pbar = tqdm()
        for iter, data in enumerate(dataloaders):
            img, label = data
            n, c, h, w = img.size()
            pbar.update(n)
            ff = torch.FloatTensor(n, linear_num).zero_().cuda()
            for i in range(2):
                if i == 1:
                    img = self.fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(input_img) 
                    ff += outputs
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            if iter == 0:
                features = torch.FloatTensor(len(dataloaders.dataset), ff.shape[1])
            start = iter * batchsize
            end = min((iter + 1) * batchsize, len(dataloaders.dataset))
            features[start:end, :] = ff
        pbar.close()
        return features
    
    def predict(self, id_person):
        with torch.no_grad():
            query_feature = self._extract_feature(
                self.model, dataloaders["query"], self.config["linear_num"], self.config["batchsize"]
            )
        for i in id_person:
            index = self._sort_img(query_feature[i], self.gallery_feature, self.gallery_label, self.gallery_cam)
            query_path, _ = image_datasets["query"].imgs[i]
            result_of_query = []
            
            # Create a figure with 1 row and 11 columns
            fig, axes = plt.subplots(1, 11, figsize=(20, 5))
            
            # Display the query image in the first subplot
            query_image = Image.open(query_path)
            axes[0].imshow(query_image)
            axes[0].set_title("Query Image")
            axes[0].axis('off')
            
            for j in range(10):
                img_path, _ = image_datasets["gallery"].imgs[index[j]]
                label = self.gallery_label[index[j]]
                result_of_query.append(label)
                
                # Display the gallery image in the subsequent subplots
                gallery_image = Image.open(img_path)
                axes[j + 1].imshow(gallery_image)
                axes[j + 1].set_title(f"Result {j + 1}")
                axes[j + 1].axis('off')
            
            most_common_result = Counter(result_of_query).most_common(1)
            id = most_common_result[0][0]
            if id == -1:
                id = 0
            
            # Show the plot
            plt.show()