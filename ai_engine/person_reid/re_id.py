from .model import classifiers
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torch.nn.functional import interpolate
import cv2
from tqdm import tqdm

class reID:
    def __init__(self, config: dict):
        self.model_path = config["model_path"]
        self.linear_num = config["linear_num"]
        self.batchsize = config["batchsize"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = classifiers
        self.model = self._load_model(self.model_path, self.backbone)

    def _load_model(self, model_path, classifiers):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            classifiers.load_state_dict(state_dict, strict=False)
            classifiers.classifier.classifier = nn.Sequential()
            classifiers.eval()
            classifiers = self.fuse_all_conv_bn(classifiers)
            return classifiers
        except Exception as e:
            raise ValueError(f"Error loading reID model: {e}")
    
    def fuse_all_conv_bn(self, model):
        stack = []
        for name, module in model.named_children():
            if list(module.named_children()):
                self.fuse_all_conv_bn(module)
                
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
            if torch.cuda.is_available():
                ff = torch.FloatTensor(n, linear_num).zero_().cuda()
            else:
                ff = torch.FloatTensor(n, linear_num).zero_()
            for i in range(2):
                if i == 1:
                    img = self.fliplr(img)
                if torch.cuda.is_available():
                    input_img = Variable(img.cuda())
                else:
                    input_img = Variable(img)
                for scale in ms:
                    if scale != 1:
                        input_img = interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
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

    @staticmethod
    def _sort_img(queryfeature, galleryfeature):
        query = queryfeature.view(-1,1)
        # print(query.shape)
        score = torch.mm(galleryfeature,query)
        score = score.squeeze(1).cpu()
        score = score.detach().numpy()
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        return index
    
    def _load_data(self):
        from .data.dataset import dataloaders, image_datasets, gallery_feature
        return dataloaders, image_datasets, gallery_feature

    async def __call__(self, id_person: list):
        dataloaders, image_datasets, gallery_feature = self._load_data()
        query_feature = self._extract_feature(
            self.model, dataloaders["query"], self.linear_num, self.batchsize
        )
        query_feature = torch.Tensor(query_feature)
        result = []
        for i in id_person:
            index = self._sort_img(query_feature[i], gallery_feature)
            query_path, _ = image_datasets["query"].imgs[i]
            
            result.append(query_path)
        return result