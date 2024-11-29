import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from utils.util import read_annotations

class XRayTrainDataset(Dataset):
    """
    X-ray 데이터셋을 위한 Dataset 클래스.

    Args:
        image_root (str): 이미지 데이터가 저장된 디렉토리 경로.
        label_root (str): JSON 라벨 데이터가 저장된 디렉토리 경로.
        fold_df (DataFrame): train/val split 정보가 포함된 데이터프레임.
        is_train (bool): train/validation 데이터 여부.
        transforms (callable, optional): Albumentations와 같은 데이터 증강 함수 (기본값: None).
        cache_data (bool): 데이터를 메모리에 캐싱할지 여부 (기본값: False).
        classes (list): 사용할 클래스 이름 리스트.

    Attributes:
        filenames (list): 이미지 파일 이름 리스트.
        labelnames (list): 라벨 파일 이름 리스트.
        class2ind (dict): 클래스 이름을 인덱스로 매핑하는 딕셔너리.
        image_dir (str): 이미지 데이터 디렉토리 경로.
        json_dir (str): JSON 라벨 데이터 디렉토리 경로.
        data_cache (dict or None): 메모리에 캐싱된 데이터를 저장하는 딕셔너리. `cache_data`가 True일 때만 사용.
    """
    def __init__(self, image_root,label_root, fold_df, is_train=True, 
                 transforms=None, cache_data=False ,classes=None):
        
        data_type = 'train' if is_train else 'val'
        
        self.filenames = fold_df[fold_df["split"]==data_type]["image_name"].tolist()
        self.labelnames = fold_df[fold_df["split"]==data_type]["json_name"].tolist()
        self.is_train = is_train
        self.transforms = transforms
        self.cache_data = cache_data
        self.classes = classes
        
        self.class2ind = dict(zip(classes, range(len(classes))))
        
        self.image_dir = image_root
        self.json_dir = label_root


        
        self.data_cache = {} if cache_data else None
        
    def __len__(self):
        """
        데이터셋의 길이를 반환.

        Returns:
            int: 데이터셋에 포함된 샘플 수.
        """
        return len(self.filenames)
    
        
    def load_data(self, img_path, label_path) -> tuple:
        """
        이미지와 라벨 데이터를 불러옴.

        Args:
            img_path (str): 이미지 파일 경로.
            label_path (str): JSON 라벨 파일 경로.

        Returns:
            tuple: (이미지 배열, JSON 'annotations' 데이터).
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = img / 255.0
        except Exception as e:
            print(f"이미지 데이터 불러오기 오류\n{e}")
            return None, None
        
        try:
            ann = read_annotations(label_path)
        except Exception as e:
            print(f'레이블링 데이터 불러오기 오류\n{e}')
            return None, None
        return img, ann
    
    def create_label_mask(self, img_shape, ann) -> np.ndarray:
        """
        이미지 크기와 'annotations' 데이터를 기반으로 라벨 마스크 생성.

        Args:
            img_shape (tuple): 이미지의 크기 (높이, 너비, 채널 수).
            ann (list): JSON의 'annotations' 데이터 리스트.

        Returns:
            np.ndarray: 생성된 라벨 마스크. 크기는 (높이, 너비, 클래스 수).
        """
        if self.classes is None:
            print("클래스 정보 없음")
            return None

        height, width = img_shape[:2]
        num_classes = len(self.classes)
        label = np.zeros((height, width, num_classes), dtype=np.uint8)
        
        for an in ann:
            class_ind = self.class2ind.get(an['label'])
            if class_ind is None:
                continue
            points = np.array(an['points'], dtype=np.int32)

            # Create a separate mask for this class
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)

            # Assign the mask to the corresponding channel
            label[..., class_ind] = mask

        return label
    
    def __getitem__(self, idx) -> tuple:
        """
        주어진 인덱스의 이미지와 라벨 데이터를 반환.

        Args:
            idx (int): 인덱스.

        Returns:
            tuple: (이미지 텐서, 라벨 텐서).
                - 이미지 텐서: 크기 (채널, 높이, 너비).
                - 라벨 텐서: 크기 (클래스 수, 높이, 너비).
        """
        image_name = self.filenames[idx]
        label_name = self.labelnames[idx]
        
        if self.cache_data and idx in self.data_cache:
            img, label = self.cache_data[idx]
        else:
            image_path = os.path.join(self.image_dir, image_name)
            label_path = os.path.join(self.json_dir, label_name)
            
            img, ann = self.load_data(image_path, label_path)
            if img is None or ann is None:
                return None,None
            label = self.create_label_mask(img.shape, ann)
            
            if self.cache_data:
                self.data_cache[idx] = (img, label)
            
        if self.transforms is not None:
            inputs = {'image': img, 'mask': label} if self.is_train else {'image':img}
            result = self.transforms(**inputs)
            img = result['image']
            label = result['mask'] if self.is_train else label
        
        img = img.transpose(2,0,1)
        label = label.transpose(2,0,1)
        
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()
        return img, label