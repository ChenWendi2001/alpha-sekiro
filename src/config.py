import argparse
import os


class Config():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        cwd = os.path.dirname(__file__)
        parser.add_argument("--obs_width", type=int, default=224) 
        parser.add_argument("--obs_height", type=int, default=224)
        parser.add_argument("--action_dim", type=int, default=4)
        parser.add_argument("--replay_capacity", type=int, default=2000)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--lr_decay", type=float, default=0.99)
        parser.add_argument('--lr_decay_every', type=int, default=200)
        parser.add_argument("--discount", type=float, default=0.98)
        parser.add_argument("--update_target_every", type=int, default=300)
        parser.add_argument("--model_dir", type=str, default=os.path.join(cwd, "checkpoints"))
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=1)
        parser.add_argument("--episodes", type=int, default=10)
        parser.add_argument("--epsilon_start", type=float, default=1)
        parser.add_argument("--epsilon_decay", type=float, default=0.995)
        parser.add_argument("--epsilon_end", type=float, default=0.1)
        parser.add_argument("--save_model_every", type=int, default=100)
        self.initialize = True
        return parser

    def parse(self):
        if not self.initialized: 
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        opt = parser.parse_args()

        self.opt = opt
        return opt


