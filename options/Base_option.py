import argparse
import os


class Base_options():
    def __init__(self):
        self.init = False

    def initialize(self):
        self.parser = argparse.ArgumentParser(description='Arguments for the training purpose.')
        # public--------------------
        self.parser.add_argument('--model', type=str, default='MODNet',
                                 choices=["MODNet", "ITMODNet"],
                                 help="training model")

    def get_args(self):
        if not self.init:
            self.initialize()
        return self.parser.parse_args()
