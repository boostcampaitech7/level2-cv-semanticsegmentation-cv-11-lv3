from .loss import CustomBCEWithLogitsLoss, FocalLoss, DiceLoss, IoULoss, CombineLoss, FocalTveskyLoss

class LossSelector():
    """
    손실 함수를 동적으로 선택 및 초기화하는 클래스.

    사용자가 새로운 손실 함수를 추가하려면:
        1. `loss` 폴더 내부에 새로운 Custom Loss 클래스 구현.
        2. 구현한 Loss Class를 이 파일(`loss_selector.py`)로 import.
        3. `self.loss_classes` 딕셔너리에 추가 (형식: "Key": LossClass).
        4. YAML 설정 파일에서 `loss_name`을 새로 추가한 키값으로 설정.

    Attributes:
        loss_classes (dict): 문자열 키와 손실 함수 클래스를 매핑.
    """
    def __init__(self) -> None:
        self.loss_classes = {
            "BCEL": CustomBCEWithLogitsLoss,
            "Focal": FocalLoss,
            "Dice": DiceLoss,
            "IoU": IoULoss,
            "FocalTve": FocalTveskyLoss,
        }

    def get_loss(self, loss_config, **loss_parameters):
        if loss_config.type == "CombineLoss":
            loss_list = [self.loss_classes[name]() for name in loss_config.loss_list]
            weights = loss_config.weights
            return CombineLoss(loss_list, weights, **loss_parameters)

        else:
            loss_name = loss_config.loss_list[0]
            return self.loss_classes.get(loss_name, None)(**loss_parameters)