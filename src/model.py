import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



class Model(nn.Module):
    def __init__(self, config):
        '''_summary_

        Args:
            config (Config): config file parsed from command args

        '''
        super(Model, self).__init__()
        raise NotImplementedError
    def forward():
        raise NotImplementedError