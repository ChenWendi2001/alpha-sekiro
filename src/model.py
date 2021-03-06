import torch
import torch.nn as nn
import logging

from resnet import resnet18, halfresnet18

def make_model(config):
    if config.model_type == "cnn":
        return CNNModel(config)
    elif config.model_type == "pose":
        return PoseModel(config)
    elif config.model_type == "fusion":
        return FusionModel(config)
    elif config.model_type == "deep":
        return DeepCNN(config)
    elif config.model_type == "half":
        return HalfCNN(config)

class CNNModel(nn.Module):
    def __init__(self, config):
        '''Init the Model

        Args:
            config (Config): config file parsed from command args

        '''
        super(CNNModel, self).__init__()
        logging.info("Initing CNN Model")
        kernel_size = (5, 5)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size, padding=(kernel_size[0] // 2, kernel_size[0] // 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
        )
        self.dense_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64 * (config.obs_width // 4) * (config.obs_height // 4), 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.action_dim),
        )
    
    def forward(self,state_tuple):
        '''forward through the model

        Args:
            state_tuple ((tensor...)): _description_

        Returns:
            tensor: q values for input states
        '''
        x = self.cnn_layers(state_tuple[0])
        
        logging.debug(x.shape)
        out = self.dense_layers(x)
        logging.debug(out.shape)
        return out


class PoseModel(nn.Module):
    def __init__(self, config):
        '''Init the Model

        Args:
            config (Config): config file parsed from command args

        '''
        super(PoseModel, self).__init__()
        logging.info("Initing Pose Model")


        self.dense_layers = nn.Sequential(
            nn.Linear(74, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.action_dim),
        )
    
    def forward(self,state_tuple):
        '''forward through the model

        Args:
            state_tuple ((tensor...)): _description_

        Returns:
            tensor: q values for input states
        '''
        x = torch.cat([state_tuple[1], state_tuple[2]], dim=-1)
        
        logging.debug(x.shape)
        out = self.dense_layers(x)
        logging.debug(out.shape)
        return out

class FusionModel(nn.Module):
    def __init__(self, config):
        '''Init the Model

        Args:
            config (Config): config file parsed from command args

        '''
        super(FusionModel, self).__init__()
        logging.info("Initing Fusion Model")

        self.cnn_layers = resnet18(pretrained=True)
        self.cnn_layers.fc = nn.Linear(512, 256)
        self.projection =  nn.Linear(74, 256)

        self.dense_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.action_dim),
        )
    
    def forward(self,state_tuple):
        '''forward through the model

        Args:
            state_tuple ((tensor...)): _description_

        Returns:
            tensor: q values for input states
        '''

        x = torch.cat([state_tuple[1], state_tuple[2]], dim=-1)
        x = self.projection(x)
        
        x = torch.cat([self.cnn_layers(state_tuple[0]).squeeze(), x], dim=-1)
        
        logging.debug(x.shape)
        out = self.dense_layers(x)
        logging.debug(out.shape)
        return out

class DeepCNN(nn.Module):
    def __init__(self, config):
        '''Init the Model

        Args:
            config (Config): config file parsed from command args

        '''
        super(DeepCNN, self).__init__()
        logging.info("Initing Deep CNN Model")

        self.cnn_layers = resnet18(pretrained=True)
        self.cnn_layers.fc = nn.Linear(512, 256)

        self.dense_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.action_dim),
        )
    
    def forward(self,state_tuple):
        '''forward through the model

        Args:
            state_tuple ((tensor...)): _description_

        Returns:
            tensor: q values for input states
        '''
        
        x = self.cnn_layers(state_tuple[0]).squeeze()
        
        logging.debug(x.shape)
        out = self.dense_layers(x)
        logging.debug(out.shape)
        return out


class HalfCNN(nn.Module):
    def __init__(self, config):
        '''Init the Model

        Args:
            config (Config): config file parsed from command args

        '''
        super(HalfCNN, self).__init__()
        logging.info("Initing Half CNN Model")

        self.cnn_layers = halfresnet18(pretrained=True)

        self.dense_layers = nn.Sequential(\
            nn.Flatten(start_dim=1),
            nn.Linear(128*28*28, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.action_dim),
        )
    
    def forward(self,state_tuple):
        '''forward through the model

        Args:
            state_tuple ((tensor...)): _description_

        Returns:
            tensor: q values for input states
        '''
        
        x = self.cnn_layers(state_tuple[0])
        
        logging.debug(x.shape)
        out = self.dense_layers(x)
        logging.debug(out.shape)
        return out