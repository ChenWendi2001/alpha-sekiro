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
        kernel_size = (5, 5)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            nn.Conv2d(32, 64, kernel_size, padding=(kernel_size[0] // 2, kernel_size[0] // 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            nn.Flatten(),
            nn.Linear(64 * (config.state_w // 2) * (config.state_h // 2), 512),
            nn.Relu(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.action_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self,state_tensor):
        '''forward through the model

        Args:
            state_tensor (tensor): _description_

        Returns:
            tensor: q values for input states
        '''
        return self.layers(state_tensor)