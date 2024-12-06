import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import albumentations as A
import pandas as pd
import argparse
from omegaconf import OmegaConf
import gc
from utils.mlflow import MLflowManager
import traceback
import matplotlib.pyplot as plt
import cv2

from utils.util import set_seed, get_classes, extract_uuid2list
from transform.transform import get_transform
from trainer.visualize import visualize_and_save_images
from data.train_dataset import XRayTrainDataset
from trainer.trainer import Trainer
from scheduler.scheduler_selector import SchedulerSelector
from loss.loss_selector import LossSelector
from models.modelselector import ModelSelector

import services.spreadsheet as sheet
import services.kakao as kakao
import services.slack as slack

def get_folds(cfg):
    '''
    지정된 폴드들의 train/val로 분리시킨 이미지 정보 csv파일을 읽어서 반환

    Args:
        cfg (OmegaConf.DictConfig): 설정 정보를 담은 객체 (폴드 리스트 및 경로 포함)

    Returns:
        folds (list): 각 폴드 데이터프레임의 리스트
    '''
    folds = []

    for num in cfg.fold_list:
        fold_path = f'fold_{num}.csv'
        fold_df = pd.read_csv(os.path.join(cfg.fold_root,fold_path))
        folds.append(fold_df)
    
    return folds
    
def main(cfg):
    '''
    메인 학습 루프. 지정된 설정에 따라 데이터 준비, 모델 생성, 학습 및 결과 로깅을 수행.

    Args:
        cfg (OmegaConf.DictConfig): 설정 정보를 담은 객체
    '''

    try:
        set_seed(cfg.seed)
        classes = get_classes()
        
        folds = get_folds(cfg)
        
        is_folds = len(folds) > 1
        
        mlflow_manager = MLflowManager(experiment_name=cfg.exp_name)
        
        uuid_list = extract_uuid2list(cfg.kakao_uuid_path)
        # kakao.send_message(uuid_list, f"{cfg.access_name}님이 서버 {cfg.server}번\n{cfg.task} epoch {cfg.max_epoch}\n학습을 시작하였습니다.")
        sheet.update_server_status(cfg.server, cfg.access_name, True, cfg.task)
        slack.send_slack_notification(f"{cfg.access_name}님이 서버 {cfg.server}번\n{cfg.task} epoch {cfg.max_epoch}\n학습을 시작하였습니다.")
        
        model_selector = ModelSelector()
        for fold_df in folds:
            cur_fold = fold_df.iloc[0]["fold"]
            print(f"--------Current Fold: {cur_fold}----------")

            model = model_selector.get_model(cfg.model)
            model = model.cuda()
            
            transform_list = [A.Resize(cfg.transform.Resize.height, cfg.transform.Resize.width)]
            transforms = get_transform(transform_list)

            train_dataset = XRayTrainDataset(
                image_root= cfg.image_root,
                label_root = cfg.label_root,
                fold_df = fold_df,
                is_train=True,
                transforms=transforms,
                cache_data=False,
                classes=classes,
            )

            val_dataset = XRayTrainDataset(
                image_root= cfg.image_root,
                label_root = cfg.label_root,
                fold_df = fold_df,
                is_train=False,
                transforms=transforms,
                cache_data=False,
                classes=classes,
            )

            train_loader = DataLoader(
                dataset=train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=8, drop_last=True
            )

            valid_loader = DataLoader(
                dataset=val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=0, drop_last=False
            )
            
            print("증강된 데이터 시각화:")
            visualize_and_save_images(train_loader, classes, save_dir="./overlay_images", max_visualizations=3)

            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler_selector = SchedulerSelector(optimizer)
            scheduler = scheduler_selector.get_scheduler(cfg.scheduler_name, **cfg.scheduler_parameter)

            loss_selector = LossSelector()
            criterion = loss_selector.get_loss(cfg.loss, **cfg.loss_parameter)

            trainer = Trainer(
                        model=model,
                        max_epoch=cfg.max_epoch,
                        train_loader=train_loader,
                        val_interval=cfg.val_interval,
                        val_loader=valid_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        save_dir=cfg.save_dir,
                        scheduler=scheduler,
                        cur_fold = cur_fold,
                        mlflow_manager = mlflow_manager,
                        run_name = cfg.run_name,
                        num_class = cfg.model.model_parameter.classes,
                        kakao_uuid_list=uuid_list,
                        access_name=cfg.access_name,
                        server=cfg.server,
                        earlystop=cfg.early_stopping
                        )

            best_dice, best_val_class = trainer.train()
            if best_dice == 0:
                continue
            else:
                dices_per_class_str = "\n".join([f"{key}: {value:.4f}" for key, value in best_val_class.items()])
                print(dices_per_class_str)
                
                if is_folds:
                    message = f"{cur_fold} 학습을 완료하였습니다."    
                else:
                    message = "학습을 완료하였습니다."
                    
                kakao.send_message(uuid_list, f"{cfg.access_name}님이 서버 {cfg.server}번\n{message}\nbest dice{best_dice}\n{dices_per_class_str}")
                slack.send_slack_notification(f"{cfg.access_name}님이 서버 {cfg.server}번\n{message}\nbest dice{best_dice}\n{dices_per_class_str}")
                
                append_data = {"model": cfg.model.model_name,
                            "scheduler": cfg.scheduler_name,
                            "optimizer": optimizer.__class__.__name__,
                            "lr": cfg.lr,
                            "epoch": cfg.max_epoch,
                            "metric": cfg.loss.type,
                            "task": cfg.task + f" ({cur_fold})",
                            "dice coef": best_dice,
                            "class score": dices_per_class_str,
                            "public score": "-"}
                
                sheet.append_training_log(cfg.access_name, append_data)
                sheet.append_training_log("Total", append_data)
                
                del model, optimizer, criterion, train_loader, valid_loader, train_dataset, val_dataset
                torch.cuda.empty_cache()
                gc.collect()
            
    except Exception as e:
        kakao.send_message(uuid_list, f"{cfg.access_name}님이 서버 {cfg.server}번\n학습도중 에러가 발생하였습니다.\n{e}")
        slack.send_slack_notification(f"{cfg.access_name}님이 서버 {cfg.server}번\n학습도중 에러가 발생하였습니다.\n{e}")
        print(e)
        print(traceback.format_exc())
    finally:
        sheet.update_server_status(cfg.server, cfg.access_name, False, cfg.task)
        print("end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base_train.yaml")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    main(cfg)

