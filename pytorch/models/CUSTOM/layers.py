import torch
import torch.nn as nn

class MultiReceptiveFieldExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiReceptiveFieldExtractor, self).__init__()
        