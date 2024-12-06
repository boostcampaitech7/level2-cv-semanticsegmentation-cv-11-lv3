import os
import time
import torch
import torch.nn as nn
import os.path as osp
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from datetime import timedelta
from torch.utils.data import DataLoader
from utils.util import save_best
from torch.cuda.amp import autocast, GradScaler
import services.kakao as kakao
import services.slack as slack
import traceback
from .Earlystopping import EarlyStopper


def dice_coef(y_true, y_pred):
    """
    Dice 계수를 계산.

    Args:
        y_true (torch.Tensor): 실제 마스크 (배치 크기 x 클래스 수 x 높이 x 너비).
        y_pred (torch.Tensor): 예측 마스크 (배치 크기 x 클래스 수 x 높이 x 너비).

    Returns:
        torch.Tensor: Dice 계수 값. 클래스별 평균 Dice 계수를 반환.
    """
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

class Trainer:
    """
    모델 학습 및 검증, Early Stopping, 모델 저장을 포함한 학습 관리를 수행하는 클래스.
    """
    def __init__(self, 
                 model: nn.Module,
                 max_epoch: int,
                 train_loader: DataLoader,
                 val_interval: int,
                 val_loader: DataLoader,
                 criterion: torch.nn.modules.loss._Loss,
                 optimizer: optim.Optimizer,
                 save_dir: str,
                 scheduler: optim.lr_scheduler,
                 cur_fold: str,
                 mlflow_manager,
                 run_name,
                 num_class,
                 kakao_uuid_list,
                 access_name,
                 server,
                 earlystop):
        """
        Trainer 초기화 함수.

        Args:
            model (nn.Module): 학습에 사용할 모델.
            max_epoch (int): 최대 학습 에폭 수.
            train_loader (DataLoader): 학습 데이터 로더.
            val_interval (int): 검증 주기 (에폭 단위).
            val_loader (DataLoader): 검증 데이터 로더.
            criterion (nn.modules.loss._Loss): 손실 함수.
            optimizer (optim.Optimizer): 옵티마이저.
            save_dir (str): 모델 저장 디렉토리.
            scheduler (optim.lr_scheduler): 학습률 스케줄러.
            cur_fold (str): 현재 폴드 이름.
            mlflow_manager: MLflow 관리 객체.
            run_name (str): MLflow 실행 이름.
            num_class (int): 클래스 수.
            kakao_uuid_list (list): 카카오 알림 대상 UUID 리스트.
            access_name (str): 학습 실행자 이름.
            server (str): 서버 번호.
            earlystop (EarlyStopper): Early Stopping 객체.
        """
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.save_dir = save_dir
        self.val_interval = val_interval
        self.cur_fold = cur_fold
        self.mlflow_manager = mlflow_manager
        self.run_name = run_name
        self.num_class = num_class
        self.kakao_uuid_list = kakao_uuid_list
        self.access_name = access_name
        self.server = server
        self.earlystop = EarlyStopper(patience=earlystop.patience, delta=earlystop.delta)
        self.scaler = GradScaler()
        

    def save_model(self, epoch, dice_score, before_path):
        """
        현재 학습 중 가장 성능이 좋은 모델 저장.

        Args:
            epoch (int): 현재 에폭 번호.
            dice_score (float): 현재 모델의 Dice 계수.
            before_path (str): 이전에 저장된 모델 경로. 있으면 삭제.

        Returns:
            str: 저장된 모델 경로.
        """
        
        if not osp.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        if before_path != "" and osp.exists(before_path):
            os.remove(before_path)

        output_path = osp.join(self.save_dir, f"{self.cur_fold}_best_{epoch}epoch_{dice_score:.4f}.pt")
        torch.save(self.model, output_path)
        return output_path


    def train_epoch(self, epoch):
        """
        한 에폭 동안 학습 수행.

        Args:
            epoch (int): 현재 에폭 번호.

        Returns:
            float: 에폭 당 평균 손실 값.
        """
        train_start = time.time()
        self.model.train()
        total_loss = 0.0

        with tqdm(total=len(self.train_loader), desc=f"{self.cur_fold}[Training Epoch {epoch}\{self.max_epoch}]", disable=False) as pbar:
            for images, masks in self.train_loader:
                images, masks = images.cuda(), masks.cuda()

                outputs = self.model(images)

                self.optimizer.zero_grad()
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, list):
                        losses = [self.criterion(output, masks) for output in outputs]
                        loss = sum(losses) / len(losses)
                    else:
                        loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        train_end = time.time() - train_start 
        print("Epoch {}, Train Loss: {:.4f} || Elapsed time: {} || ETA: {}\n".format(
            epoch,
            total_loss / len(self.train_loader),
            timedelta(seconds=train_end),
            timedelta(seconds=train_end * (self.max_epoch - epoch))
        ))
        return total_loss / len(self.train_loader)
    

    def validation(self, epoch):
        """
        현재 모델을 검증 데이터셋에서 평가.

        Args:
            epoch (int): 현재 에폭 번호.

        Returns:
            tuple: (평균 Dice 계수, 클래스별 Dice 계수 딕셔너리, 평균 검증 손실 값)
        """
        torch.cuda.empty_cache()
        val_start = time.time()
        self.model.eval()

        total_loss = 0
        dices = []

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'{self.cur_fold}[Validation Epoch {epoch}]', disable=False) as pbar:
                for images, masks in self.val_loader:
                    images, masks = images.cuda(), masks.cuda()

                    outputs = self.model(images)
                    # outputs = self.model(images)['out']
                    
                    if isinstance(outputs, list):  # Deep supervision 처리
                        avg_output = None
                        total_outputs = 0
                        losses = []

                        for output in outputs:
                            if output.size()[-2:] != masks.size()[-2:]:
                                output = F.interpolate(output, size=masks.shape[-2:], mode="bilinear")
                            
                            losses.append(self.criterion(output, masks))
                            
                            if avg_output is None:
                                avg_output = torch.sigmoid(output)
                            else:
                                avg_output += torch.sigmoid(output)
                            total_outputs += 1
                        
                        outputs = avg_output / total_outputs
                        loss = sum(losses) / len(losses)
                    else:
                        if outputs.size()[-2:] != masks.size()[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")
                        loss = self.criterion(outputs, masks)
                        outputs = torch.sigmoid(outputs)
                    total_loss += loss.item()

                    dice = dice_coef(outputs, masks)
                    dices.append(dice)

                    pbar.update(1)
                    pbar.set_postfix(dice=torch.mean(dice).item(), loss=loss.item())

        val_end = time.time() - val_start
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(self.val_loader.dataset.class2ind, dices_per_class)
        ]

        dice_str = "\n".join(dice_str)
        print(dice_str)
        
        avg_dice = torch.mean(dices_per_class).item()
        print("avg_dice: {:.4f}".format(avg_dice))
        print("Validation Loss: {:.4f} || Elapsed time: {}\n".format(
            total_loss / len(self.val_loader),
            timedelta(seconds=val_end),
        ))

        class_dice_dict = {f"{c}" : d for c, d in zip(self.val_loader.dataset.class2ind, dices_per_class)}
        
        return avg_dice, class_dice_dict, total_loss / len(self.val_loader)
    
    def train(self):
        print(f'Start training..')
        best_dice = 0.
        best_val_class = dict()
        best_val_loss = 0.
        try:
            with self.mlflow_manager.start_run(run_name=self.run_name):
                    self.mlflow_manager.log_params({
                        "num_epoch": self.max_epoch,
                        "val_step": self.val_interval,
                        "n_classes": self.num_class,
                        "save_dir": self.save_dir,
                        "optimizer": self.optimizer.__class__.__name__,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
            
                    for epoch in range(1, self.max_epoch + 1):
                        
                        train_loss = self.train_epoch(epoch)
                        self.mlflow_manager.log_metrics(metrics={"train_loss":train_loss},step=epoch)
                        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
                        if epoch % self.val_interval == 0:
                            avg_dice, dices_per_class, val_loss = self.validation(epoch)
                            self.mlflow_manager.log_metrics({"val_loss":val_loss}, step=epoch)
                            self.mlflow_manager.log_metrics({"val_dice":avg_dice}, step=epoch)
                            
                            if best_dice < avg_dice:
                                print(f"Best performance at epoch: {epoch}, {best_dice:.4f} -> {avg_dice:.4f}\n")
                                best_dice = avg_dice
                                best_val_class = dices_per_class
                                best_val_loss = val_loss
                                self.mlflow_manager.log_metrics({"best_dice":best_dice}, step=epoch)
                                save_best(self.model, self.save_dir, cur_fold=self.cur_fold)
                                
                            self.earlystop(avg_dice)
                            if self.earlystop.early_stop:
                                print(f"Early Stopping at epoch {epoch} with best dice : {best_dice:.4f}")
                                break
                                
                        if self.max_epoch >= 3 and epoch % (self.max_epoch // 3) == 0:
                            dices_per_class_str = "\n".join([f"{key}: {value:.4f}" for key, value in best_val_class.items()])
                            message = f'서버 {self.server}번 {self.access_name}님의\n학습 현황 epoch {epoch}\nbest dice score : {best_dice}\n{dices_per_class_str}'
                            kakao.send_message(self.kakao_uuid_list, message)
                            slack.send_slack_notification(message=message)
                            
                        self.scheduler.step(val_loss)
        except Exception as e:
            error_message = (
            f"서버 {self.server}번 {self.access_name}님의 학습 중 에러 발생\nError: {str(e)}"
            )
            print(traceback.format_exc())
            kakao.send_message(self.kakao_uuid_list, error_message)
            slack.send_slack_notification(f"서버 {self.server}번 {self.access_name}님의 학습 중 에러 발생\nError: {str(e)}")
        return best_dice, best_val_class