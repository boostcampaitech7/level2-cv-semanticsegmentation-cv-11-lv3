from .DUCKNet.duck_net import DUCKNet
from .UNet.unet import UNet
from .NestedUNet.nestedunet import NestedUNet
from .UNet3plus.unet3plus import UNet3Plus
from .fcn_resnet50.fcn_resnet50 import fcn_resnet
from .RAPUNet.rapunet import RAPUNet
from .SegFormer.SegFormer import SegFormer

class ModelSelector():
    def __init__(self):
        self.model_list = {
            "UNet": UNet,
            "DUCKNet": DUCKNet,
            "fcn_resnet50": fcn_resnet,
            "NestedUNet": NestedUNet,
            "UNet3Plus": UNet3Plus,
            "RAPUNet": RAPUNet,
            "SegFormer": SegFormer,
        }
        
    def get_model(self, model_cfg):
        model_name = model_cfg.get("model_name")
        model_params = model_cfg.get("model_parameter")
        
        if model_name not in self.model_list:
            raise ValueError(f"사용가능한 모델: {list(self.model_list.keys())}")
        
        model_class = self.model_list[model_name]
        print(model_class)

        if model_name == 'UNet':
            in_channels = model_params.get("in_channels", 3)
            num_classes = model_params.get("classes", 29)
            model = model_class(in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'DUCKNet':
            in_channels = model_params.get("in_channels", 3)
            num_classes = model_params.get("classes", 29)
            starting_filters = model_params.get("start_filters", 17)
            model = model_class(in_channels=in_channels, num_classes=num_classes, starting_filters=starting_filters)
        elif model_name == "NestedUNet":
            in_channels = model_params.get("in_channels", 3)
            num_classes = model_params.get("classes", 29)
            deep_supervision = model_params.get("deep_supervision", False)
            model = model_class(in_channels=in_channels, num_classes=num_classes, deep_supervision=deep_supervision)
        elif model_name == "UNet3Plus":
            in_channels = model_params.get("in_channels", 3)
            num_classes = model_params.get("classes", 29)
            deep_supervision = model_params.get("deep_supervision", False)
            cgm = model_params.get("cgm", False)
            model = model_class(in_channels=in_channels, num_classes=num_classes, deep_supervision=deep_supervision, cgm=cgm)
        elif model_name == 'fcn_resnet50':
            num_classes = model_params.get("classes", 29)
            model_init = model_class(num_classes=num_classes)
            model = model_init.get_model()
        elif model_name == "RAPUNet":
            in_channels = model_params.get("in_channels", 3)
            num_classes = model_params.get("classes", 29)
            starting_filters = model_params.get("start_filters", 17)
            model = model_class(in_channels=in_channels, num_classes=num_classes, starting_filters=starting_filters)
        elif model_name =='SegFormer':
            num_classes=model_params.get("classes",29)
            model=model_class(in_channels=in_channels ,num_classes=num_classes)
        return model
            
        