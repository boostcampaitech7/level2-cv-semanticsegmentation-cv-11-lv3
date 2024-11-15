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

from utils.util import set_seed, find_file, get_classes, save_ckpt, save_best
from transform.transform import get_transform
from data.train_dataset import XRayTrainDataset
from data.test_dataset import XRayInferenceDataset
from loss.loss import combine_loss
from trainer.trainer import Trainer
from scheduler.scheduler_selector import SchedulerSelector
from loss.loss_selector import LossSelector
# from utils.util import inference_save, inference_to_csv
# from train.test import test

# 1. 하이퍼파라미터 설정

# IND2CLASS = {i: cls for i, cls in enumerate(CLASSES)}

def initialize_model(num_classes):
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model



def get_folds(cfg):
    folds = []

    for num in cfg.fold_list:
        fold_path = f'fold_{num}.csv'
        fold_df = pd.read_csv(os.path.join(cfg.fold_root,fold_path))
        folds.append(fold_df)
    
    return folds
    


def main(cfg):
    set_seed(cfg.seed)
    classes = get_classes()

    folds = get_folds(cfg)
    mlflow_manager = MLflowManager(experiment_name=cfg.exp_name)

    for fold_df in folds:
        fold_num = fold_df.iloc[0]["fold"]
        print(f"--------Current Fold: {fold_num}----------")

        model = initialize_model(cfg.model_parameter.classes)
        model = model.cuda()
        transform_list = [A.Resize(512, 512)]
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

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-6)
        scheduler_selector = SchedulerSelector(optimizer)
        scheduler = scheduler_selector.get_scheduler(cfg.scheduler_name, **cfg.scheduler_parameter)

        # criterion = combine_loss
        loss_selector = LossSelector()
        criterion = loss_selector.get_loss(cfg.loss_name, **cfg.loss_parameter)


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
                    fold_num = fold_num,
                    mlflow_manager = mlflow_manager,
                    run_name = cfg.run_name,
                    num_class = cfg.model_parameter.classes,
                    )


        trainer.train()

        del test_loader, test_dataset, model, optimizer, criterion, train_loader, valid_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()




        test_pngs = find_file(os.path.join(cfg.test_root, "test/DCM"), ".png")

        test_dataset = XRayInferenceDataset(
            pngs=test_pngs,
            root_dir= cfg.test_root,
            transforms=transforms
        )

        test_loader = DataLoader(
            dataset=test_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=0, drop_last=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base_train.yaml")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    main(cfg)

