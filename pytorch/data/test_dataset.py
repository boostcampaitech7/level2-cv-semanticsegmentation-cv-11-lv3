import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

class XRayInferenceDataset(Dataset):
    """
    X-Ray 이미지를 처리하기 위한 PyTorch Dataset 클래스.
    테스트 데이터셋을 로드하고 전처리 및 변환을 수행.

    Args:
        pngs (list of str): 테스트 이미지 파일 이름 리스트.
        root_dir (str): 데이터 루트 디렉토리 경로.
        transforms (callable, optional): 이미지를 변환할 함수 또는 객체 (기본값: None).
    """
    def __init__(self, pngs, root_dir, transforms=None): 
        """
        Dataset 초기화 함수.

        Args:
            pngs (list of str): 테스트 이미지 파일 이름 리스트.
            root_dir (str): 데이터 루트 디렉토리 경로.
            transforms (callable, optional): 이미지를 변환할 함수 또는 객체 (기본값: None).
        """
        self.pngs = pngs
        self.root_dir = root_dir
        
        filenames = pngs
        filenames = np.array(sorted(filenames))
        
        self.filenames = filenames
        self.transforms = transforms
        
        self.test_image_root = root_dir + "/test/DCM"
    
    def __len__(self):
        """
        데이터셋 크기 반환.

        Returns:
            int: 데이터셋에 포함된 이미지 수.
        """
        return len(self.filenames)
    
    def __getitem__(self, item):
        """
        주어진 인덱스의 이미지와 파일 이름 반환.

        Args:
            item (int): 가져올 데이터의 인덱스.

        Returns:
            tuple: (torch.Tensor, str) 
                - torch.Tensor: 전처리된 이미지 (C x H x W 형식).
                - str: 이미지 파일 이름.
        """
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