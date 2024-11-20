import os
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from trainer.test import test

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data.test_dataset import XRayInferenceDataset
from utils.util import inference_save, inference_to_csv, find_file
from datetime import datetime
from utils.mlflow import MLflowManager



if __name__=="__main__":
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model to use")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home")
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--resize", type=int, default=1024, help="Size to resize images (both width and height)")
    args = parser.parse_args()

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

    # Inference and save results
    rles, filename_and_class = test(args, test_loader, thr=args.thr)
    save_dir = f"./inference_results/{start_time}"
    
    result_df = inference_to_csv(filename_and_class, rles, path=save_dir)
    image_root = os.path.join(args.image_root,'test/DCM')

    inference_save(filename_and_class, image_root, result_df=result_df, save_dir=save_dir)
    
    mlflowmanager=MLflowManager(experiment_name='Inference_Output')
    with mlflowmanager.start_run(run_name=f"Inference_{start_time}"):
        mlflowmanager.log_params({"Resize":args.resize,"Threshold":args.thr,"Model_path":args.model})

        if os.path.exists(save_dir):
    
            mlflowmanager.log_artifacts(save_dir) 
        else:
            print(f"디렉토리 경로 {save_dir} 가 없습니다")
        
        print(f"결과가 MLflow Inference_Output {start_time}에 저장되었습니다.")

