import argparse
import os
from secrets import choice


class Config():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        cwd = os.path.dirname(__file__)
        parser.add_argument("--obs_width", type=int, default=224) 
        parser.add_argument("--obs_height", type=int, default=224)
        parser.add_argument("--action_dim", type=int, default=7)
        parser.add_argument("--replay_capacity", type=int, default=2000)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--lr_decay", type=float, default=0.99)
        parser.add_argument('--lr_decay_every', type=int, default=100)
        parser.add_argument("--discount", type=float, default=0.95)
        parser.add_argument("--update_target_every", type=int, default=50)
        parser.add_argument("--model_dir", type=str, default=os.path.join(cwd, '..', "checkpoints"))
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--episodes", type=int, default=100)
        parser.add_argument("--epsilon_start", type=float, default=1)
        parser.add_argument("--epsilon_decay", type=float, default=0.999)
        parser.add_argument("--epsilon_end", type=float, default=0.05)
        parser.add_argument("--save_model_every", type=int, default=20)
        parser.add_argument("--log_dir", type=str, default=os.path.join(cwd, '..', "logs"))
        parser.add_argument("--load_ckpt", action="store_true")
        parser.add_argument("--model_name", type=str, default="")
        parser.add_argument("--ckpt_name", type=str)
        parser.add_argument("--test_mode", action="store_true")
        parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "pose", "fusion"])

        self.initialize = True
        return parser

    def parse(self):
        if not self.initialized: 
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        opt = parser.parse_args()
        if opt.test_mode and not opt.load_ckpt:
            print("need checkpoints to test!")
            raise RuntimeError()
        self.opt = opt
        return opt


