import os
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F
from utils.util import get_IND2CLASS
from trainer.test_rle import encode_mask_to_rle
from trainer.test import batch_soft_voting

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data.test_dataset import XRayInferenceDataset
from utils.util import inference_save, inference_to_csv, find_file
from datetime import datetime

if __name__=="__main__":
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder_path", type=str, help="The folder path where the model files are located")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home")
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--resize", type=int, default=1024, help="Size to resize images (both width and height)")
    args = parser.parse_args()

    model_list = []
    for root, dirs, files in os.walk(args.model_folder_path):
        for file in files:
            if file.endswith(".pt"):
                model_list.append(os.path.join(root, file))

    print(f"model_list: {model_list}")

    total_output = []

    test_pngs = find_file(os.path.join(args.image_root,'test/DCM'), ".png")

    tf = A.Resize(height=args.resize, width=args.resize)

    test_dataset = XRayInferenceDataset(test_pngs,
                                        args.image_root,
                                        transforms=tf)

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    rles, filename_and_class = batch_soft_voting(test_loader,model_list)

    save_dir = f"./inference_results/{start_time}"
    os.makedirs(save_dir, exist_ok=True)

    result_df = inference_to_csv(filename_and_class, rles,path=save_dir)

    image_root = os.path.join(args.image_root,'test/DCM')

    inference_save(filename_and_class, image_root, image_size=args.resize, result_df=result_df, save_dir=save_dir)


