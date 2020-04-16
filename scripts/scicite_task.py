"""
Task specification for the scicite task, as required by the framework
It includes task-specific functions defined in the class 
and task-specific parameters read from the json config file.

It is the main file to run the experiment (not the framework.py file).

Usage:
    python scicite_task.py [experiment_dir] [config_json_file]

"""

from framework import Experiment, DataIterator
import sys
import os
import json
import utils
from pathlib import Path

class SciciteTask:
    def __init__(self, exp_num, cfg_json):
        self.exp_num = exp_num

        self.cfg = {}
        for k, v in json.load(open(cfg_json)).items():
            self.cfg[k] = v

        # directories
        self.root = Path(__file__).parent.parent.absolute()
        self.exp_dir = f'{self.root}/exp/exp-{exp_num}'
        self.script_dir = f'{self.root}/scripts'

        # the actual SciBERT vectors, all in one file
        self.data_path = f'{self.root}/data/all.src.npz' 

        # indices of the train (seed) data, extra data, dev and test data
        self.train_path = f'{self.exp_dir}/data/seed.idx'
        self.extra_path = f'{self.exp_dir}/data/extra.idx'
        self.dev_path = f'{self.root}/data/dev.idx'
        self.test_path = f'{self.root}/data/test.idx'
        
        self.exp = Experiment(self)

    def start(self):
        """
        Run the full experiments
        """
        self.prepare()
        self.exp.start()

    def make_seed_data(self):
        os.makedirs(f'{self.exp_dir}/data', exist_ok=True)

        all_train_data = open(f'{self.root}/data/train.idx').readlines()
        # only take the first N instances as the seed training data,
        # and treat the rest as extra data
        with open(self.train_path, 'w') as f,\
             open(self.extra_path, 'w') as g:
            for i in range(self.cfg['num_seed_data']):
                f.write(all_train_data[i])
            for i in range(self.cfg['num_seed_data'], len(all_train_data)):
                g.write(all_train_data[i])

    def prepare(self):
        """
        Prepare extra data, select vocabulary, etc.
        """
        
        # create seed data and extra data from the original training data
        self.make_seed_data()

        # initialize the extra data iterator
        # original data
        src = self.get_src_from_file(self.train_path)
        src = src[:self.cfg['num_extra_data']]

        # extra data
        extra = task.get_src_from_file(self.extra_path)

        self.exp.data_iterator = DataIterator(extra, set(src))


    def get_src_from_file(self, filename):
        # the first column is index
        return [int(line.strip().split('\t')[0]) for line in open(filename)]

    def get_tgt_from_file(self, filename):
        # the (optional) second column is the label
        return [line.strip().split('\t')[1] for line in open(filename)]

    def get_confidence_from_file(self, filename):
        # the (optional) third column is confidence of the classifier
        return [float(line.strip().split('\t')[2]) for line in open(filename)]




if __name__ == '__main__':
    task = SciciteTask(*sys.argv[1:])
    task.start()