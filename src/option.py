import argparse
import os


class Config():
    def __init__(self, type):
        self.initialized = False

    def initialize(self, parser):
        cwd = os.path.dirname(__file__)
        parser.add_argument("--obs_width", required=True) 
        parser.add_argument("--obs_height", required=True)
        parser.add_argument("--action_dim", required=True)
        parser.add_argument("--replay_capacity", type=int, default=2000)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--lr_decay", type=float, default=0.99)
        parser.add_argument('--lr_decay_every', type=int, default=200)
        parser.add_argument("--discount", type=float, default=0.98)
        parser.add_argument("--update_target_every", type=int, default=300)
        parser.add_argument("--model_dir", type=str, default=os.path.join(cwd, "checkpoints"))
        parser.add_argument("--batch_size", type=int, default=64)
        self.initialize = True
        return parser

    def parse(self):
        if not self.initialized: 
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        opt = parser.parse_args()

        self.opt = opt
        return opt


