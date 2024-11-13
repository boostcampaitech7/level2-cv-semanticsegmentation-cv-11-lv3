import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from utils.util import read_annotations

class XRayTrainDataset(Dataset):
    '''
    X-ray 데이터셋을 위한 Dataset 클래스
    
    Args:
        root_dir (str): 데이터셋의 루트 경로
        pngs (list): PNG 이미지 파일의 리스트
        jsons (list): JSON 라벨 파일의 리스트
        is_train (bool): train/validation 여부
        transform (albumentations.Compose): 이미지와 마스크에 적용할 증강
        cache_data (bool): 데이터를 메모리에 캐싱할지 여부
        n (int): GroupKFold의 n_split값
        classes (list): 사용할 클래스의 리스트
    
    Attributes:
        class2ind (dict): 클래스 이름을 인덱스로 매핑하는 딕셔너리
        ind2class (dict): 인덱스를 클래스 이름으로 매핑하는 딕셔너리
        data_cache (dict): 캐싱할 데이터를 저장하는 딕셔너리
    '''
    def __init__(self, root_dir, fold_df, is_train=True, 
                 transforms=None, cache_data=False ,classes=None):
        
        data_type = 'train' if is_train else 'val'
        
        self.root_dir = root_dir
        self.filenames = fold_df[fold_df["split"]==data_type]["image_name"].tolist()
        self.labelnames = fold_df[fold_df["split"]==data_type]["json_name"].tolist()
        self.is_train = is_train
        self.transforms = transforms
        self.cache_data = cache_data
        self.classes = classes
        
        self.class2ind = dict(zip(classes, range(len(classes))))
        
        self.image_dir = root_dir + "/train/DCM" 
        self.json_dir = root_dir + "/train/outputs_json"


        
        self.data_cache = {} if cache_data else None
        
    def __len__(self):
        '''
        데이터셋의 길이 반환
        '''
        return len(self.filenames)
    
        
    def load_data(self, img_path, label_path) -> tuple:
        """
        이미지와 라벨 데이터를 불러옵니다

        Args:
            img_path (str): 이미지 파일 경로
            label_path (str): JSON 라벨 파일 경로

        Returns:
            tuple: (이미지 배열, 'annotations' 데이터)
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
        주어진 이미지 크기와 'annotations' 데이터를 기반으로 라벨 마스크를 생성

        Args:
            img_shape (tuple): 이미지의 크기
            ann (list): 'annotations' 데이터 리스트

        Returns:
            np.ndarray: 생성된 라벨 마스크
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
        주어진 인덱스의 이미지와 라벨 데이터를 반환

        Args:
            idx (int): 인덱스

        Returns:
            tuple: (이미지 텐서, 라벨 텐서)
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