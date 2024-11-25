from .loss import CustomBCEWithLogitsLoss, FocalLoss, DiceLoss, IoULoss, CombineLoss, FocalTveskyLoss

class LossSelector():
    """
    loss를 새롭게 추가하기 위한 방법
        1. loss 폴더 내부에 사용하고자하는 custom loss 구현
        2. 구현한 Loss Class를 loss_selector.py 내부로 import
        3. self.loss_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 loss_name을 설정한 key값으로 변경
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