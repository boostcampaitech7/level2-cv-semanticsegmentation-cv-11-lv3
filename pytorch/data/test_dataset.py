import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

class XRayInferenceDataset(Dataset):
    def __init__(self, pngs, root_dir, transforms=None):
        self.pngs = pngs
        self.root_dir = root_dir
        
        filenames = pngs
        filenames = np.array(sorted(filenames))
        
        self.filenames = filenames
        self.transforms = transforms
        
        self.test_image_root = root_dir + "/test/DCM"
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.test_image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name