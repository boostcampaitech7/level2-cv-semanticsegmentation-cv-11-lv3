from .DUCKNet.duck_net import DUCKNet
from .UNet.unet import UNet
from .fcn_resnet50.fcn_resnet50 import fcn_resnet

class ModelSelector():
    # def __init__(self, model_name, in_channels, num_classes, starting_filters=None):

    def __init__(self):
        self.model_list = {
            "UNet": UNet,
            "DUCKNet": DUCKNet,
            "fcn_resnet50": fcn_resnet
        }
        
    def get_model(self, model_cfg):
        model_name = model_cfg.get("model_name")
        model_params = model_cfg.get("model_parameter")
        
        if model_name not in self.model_list:
            raise ValueError(f"사용가능한 모델: {list(self.model_list.keys())}")
        
        model_class = self.model_list[model_name]
        
        if model_name == 'UNet':
            in_channels = model_params.get("in_channels", 3)
            num_classes = model_params.get("classes", 29)
            model = model_class(in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'DUCKNet':
            in_channels = model_params.get("in_channels", 3)
            num_classes = model_params.get("classes", 29)
            starting_filters = model_params.get("start_filters", 17)
            model = model_class(in_channels=in_channels, num_classes=num_classes, starting_filters=starting_filters)
        elif model_name == 'fcn_resnet50':
            num_classes = model_params.get("classes", 29)
            model_init = model_class(num_classes=num_classes)
            model = model_init.get_model()
        
        return model
            
        