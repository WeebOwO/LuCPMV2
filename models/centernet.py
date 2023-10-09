import torch.nn as nn

from .backbone import resnet18
from .loss import Detection_loss

class CPMNetV2(nn.Module):
    def __init__(self, config, device):
        super(CPMNetV2, self).__init__()
        self.loss = Detection_loss(crop_size=config['crop_size'], topk=config['topk'], spacing=config['spacing'])
        self.backbone = resnet18(detection_loss=self.loss, device=device)

    def forward(self, x):
        return self.backbone(x)

def build_detector(config, device):
    return CPMNetV2(config, device)
