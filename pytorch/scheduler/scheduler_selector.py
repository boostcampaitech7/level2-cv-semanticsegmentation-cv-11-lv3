from torch.optim import lr_scheduler
from .CustomCAWR.CustomCosineAnnealingWarmupRestarts import CustomCosineAnnealingWarmupRestarts
from torch.optim import lr_scheduler
from .CustomCAWR.CustomCosineAnnealingWarmupRestarts import CustomCosineAnnealingWarmupRestarts

def multi_step_lr(optimizer, **scheduler_parameter):
    '''
    MultiStepLR 스케줄러 생성
    
    Args:
        optimizer (torch.optim.Optimizer): 옵티마이저
        **scheduler_parameter: 스케줄러 파라미터
    
    Returns:
        torch.optim.lr_scheduler.MultiStepLR: 생성된 스케줄러
    '''
    return lr_scheduler.MultiStepLR(optimizer, **scheduler_parameter)

def cosine_annealing_lr(optimizer, **scheduler_parameter):
    '''
    CosineAnnealingLR 스케줄러 생성
    
    Args:
        optimizer (torch.optim.Optimizer): 옵티마이저
        **scheduler_parameter: 스케줄러 파라미터
    
    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR: 생성된 스케줄러
    '''
    return lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_parameter)

def cosine_annealing_warm_restarts(optimizer, **scheduler_parameter):
    '''
    CosineAnnealingWarmRestarts 스케줄러 생성
    
    Args:
        optimizer (torch.optim.Optimizer): 옵티마이저
        **scheduler_parameter: 스케줄러 파라미터
    
    Returns:
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts: 생성된 스케줄러
    '''
    return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_parameter)

def custom_cosine_annealing_warmup_restarts(optimizer, **scheduler_parameter):
    '''
    CustomCosineAnnealingWarmupRestarts 스케줄러 생성
    
    Args:
        optimizer (torch.optim.Optimizer): 옵티마이저
        **scheduler_parameter: 스케줄러 파라미터
    
    Returns:
        CustomCosineAnnealingWarmupRestarts: 생성된 스케줄러
    '''
    return CustomCosineAnnealingWarmupRestarts(optimizer=optimizer, **scheduler_parameter)

def reduce(optimizer, **scheduler_parameter):
    '''
    ReduceLROnPlateau 스케줄러 생성
    
    Args:
        optimizer (torch.optim.Optimizer): 옵티마이저
        **scheduler_parameter: 스케줄러 파라미터
    
    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: 생성된 스케줄러
    '''
    return lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **scheduler_parameter)

def reduce(optimizer, **scheduler_parameter):
    '''
    ReduceLROnPlateau 스케줄러 생성
    
    Args:
        optimizer (torch.optim.Optimizer): 옵티마이저
        **scheduler_parameter: 스케줄러 파라미터
    
    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: 생성된 스케줄러
    '''
    return lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **scheduler_parameter)
class SchedulerSelector():
    """
    scheduler를 새롭게 추가하기 위한 방법
        1. torch에서 제공하는 scheduler는 scheduler_selector.py에 함수로 구현
        2. 직접 구현해야하는 scheduler는 scheduler 폴더 내부에 class로 구현
        2. 구현한 Scheduler Class를 scheduler_selector.py 내부로 import
        3. self.scheduler_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 scheduler_name을 설정한 key값으로 변경
    """
    def __init__(self, optimizer) -> None:
        self.scheduler_classes = {
            "MultiStepLR" : multi_step_lr,
            "CosineAnnealingLR" : cosine_annealing_lr,
            "CosineAnnealingWR" : cosine_annealing_warm_restarts,
            "CustomCAWR" : custom_cosine_annealing_warmup_restarts,
            "Reduce" : reduce
        }
        self.optimizer = optimizer

    def get_scheduler(self, scheduler_name, **scheduler_parameter):
        '''
        스케줄러 생성
        
        Args:
            scheduler_name (str): 스케줄러 이름
            **scheduler_parameter: 스케줄러 파라미터
        
        Returns:
            스케줄러 객체
        '''
        return self.scheduler_classes.get(scheduler_name, None)(self.optimizer, **scheduler_parameter)
