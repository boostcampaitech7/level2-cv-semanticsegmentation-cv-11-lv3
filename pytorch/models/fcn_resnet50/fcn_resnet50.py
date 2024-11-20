import torch.nn as nn
from torchvision import models


class fcn_resnet():
    def __init__(self, num_classes):
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def get_model(self):
        return self.model