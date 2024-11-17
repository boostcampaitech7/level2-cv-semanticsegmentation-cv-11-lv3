'''
main.py 수정 필요
'''

















# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import models
# import albumentations as A
# import pandas as pd

# from utils.util import set_seed, find_file, save_ckpt, save_best
# from transform.transform import get_transform
# from data.train_dataset import XRayTrainDataset
# from data.test_dataset import XRayInferenceDataset
# from loss.loss import combine_loss
# from train.train import train, validation
# from utils.util import inference_save, inference_to_csv
# from train.test import test
# from utils.mlflow import MLflowManager
# # 1. 하이퍼파라미터 설정
# exp_name='hi'
# run_name='test'
# ROOT_PATH = "/data/ephemeral/home"
# FOLD_PATH = "/data/ephemeral/home/folds"
# BATCH_SIZE = 16
# LR = 5e-4
# RANDOM_SEED = 21
# NUM_EPOCHS = 1
# VAL_EVERY = 1
# SAVE_DIR = "checkpoints"
# FOLDNUM = 2

# CLASSES = [
#     'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
#     'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
#     'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
#     'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
#     'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
#     'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
# ]
# CLASS2IND = {cls: i for i, cls in enumerate(CLASSES)}
# IND2CLASS = {i: cls for i, cls in enumerate(CLASSES)}

# set_seed(RANDOM_SEED)

# fold_path = f'fold_{FOLDNUM}.csv'
# fold_df = pd.read_csv(os.path.join(FOLD_PATH,fold_path))


# # train_pngs = sorted(find_file(os.path.join(ROOT_PATH, "train/DCM"), ".png"))
# # train_jsons = sorted(find_file(os.path.join(ROOT_PATH, "train/outputs_json"), ".json"))
# test_pngs = find_file(os.path.join(ROOT_PATH, "test/DCM"), ".png")


# print(f"Number of JSON files: {len(test_pngs)}")




# transform_list = [A.Resize(512, 512)]
# transforms = get_transform(transform_list)




# train_dataset = XRayTrainDataset(
#     root_dir=ROOT_PATH,
#     fold_df = fold_df,
#     is_train=True,
#     transforms=transforms,
#     cache_data=False,
#     classes=CLASSES,
# )

# valid_dataset = XRayTrainDataset(
#     root_dir=ROOT_PATH,
#     fold_df = fold_df,
#     is_train=False,
#     transforms=transforms,
#     cache_data=False,
#     classes=CLASSES,
# )

# test_dataset = XRayInferenceDataset(
#     pngs=test_pngs,
#     root_dir=ROOT_PATH,
#     transforms=transforms
# )

# train_loader = DataLoader(
#     dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True
# )

# valid_loader = DataLoader(
#     dataset=valid_dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False
# )

# test_loader = DataLoader(
#     dataset=test_dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False
# )

# def initialize_model(num_classes):
#     model = models.segmentation.fcn_resnet50(pretrained=True)
#     model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
#     return model

# model = initialize_model(len(CLASSES))
# model = model.cuda()

# criterion = combine_loss
# optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

# def main():
#     print(f"FOLD NUMBER: FOLD_{FOLDNUM}")
#     print("학습 시작...")
#     best_dice = 0.0
#     mlflow_manager =MLflowManager(experiment_name=exp_name)
#     train(
#         model=model,
#         num_epoch=NUM_EPOCHS,
#         data_loader=train_loader,
#         val_step=VAL_EVERY,
#         val_loader=valid_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         save_dir=SAVE_DIR,
#         classes=CLASSES,
#         mlflow_manager=mlflow_manager,
#         run_name=run_name
#     )

#     print("학습 완료.")
#     # best_model_path = os.path.join(SAVE_DIR, 'best.pt')
#     # model = torch.load(best_model_path)
    
#     # print("테스트 데이터로 추론 시작...")
#     # rles, filename_and_class = test(model, test_loader, IND2CLASS)

#     # inference_save(rles, filename_and_class, os.path.join(ROOT_PATH, "test/DCM"), CLASSES) # 현재 오류 있음
#     # inference_to_csv(filename_and_class, rles)

# if __name__ == "__main__":
#     main()
