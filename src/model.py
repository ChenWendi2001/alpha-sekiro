import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, config):
        '''Init the Model

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
            nn.Linear(64 * (config.obs_width // 4) * (config.obs_height // 4), 512),
            nn.Relu(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.action_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self,state_tuple):
        '''forward through the model

        Args:
            state_tuple ((tensor...)): _description_

        Returns:
            tensor: q values for input states
        '''
        return self.layers(state_tuple[0])