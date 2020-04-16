"""
The main framework of the ensemble self-learning method.
It contains the generic funcionalities, and expects some task-specific 
functions implemented by the Task class

General procedure (iteratively):
    (1) train N models, add them into a model pool
    (2) find K models from the pool that maximizes the accuracy on the dev set 
        (using genetic algorithm here)
    (3) use the found combination to predict a batch of unlabeled data,
        and select the instances with high agreement as additional training data

"""


import sys, os
import re
import json
import glob
import subprocess
import time
from itertools import *
from collections import defaultdict
import numpy as np
import multiprocessing as mp
import time
import random
from shutil import copyfile
import utils 


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def log(self, cat, msg):
        with open('%s/%s.log' % (self.log_dir, cat), 'a+') as f:
            f.write(time.strftime("[%Y-%m-%d %H:%M:%S]\n", time.localtime()))
            f.write(msg + '\n\n')
        print(msg)

class DataIterator(object):
    """
    Write a chunk of the overall data into a file, 
    for the external training script to use
    """


    def __init__(self, extra_data, original_data=[]):
        self.selected = set(original_data)
        self.data = extra_data


    def iterate_data(self):
        while True:
            random.shuffle(self.data)
            yield from self.data

    def get_next(self, output_file, num = 10000):
        '''write the next N instances into a file and return the file name
        file is always overwriten to avoid unnecessary space usage
        '''
        count = 0
        with open(output_file, 'w') as out: 
            for inst in self.iterate_data():
                if inst not in self.selected:
                    count += 1
                    out.write(f'{str(inst)}\n')
                    if count == num:
                        return



class Experiment(object):
    def __init__(self, task):
        self.task = task
        self.cfg = task.cfg
        self.root = task.root
        self.exp_dir = task.exp_dir
        self.script_dir = task.script_dir
        self.train_path = task.train_path
        self.dev_path = task.dev_path
        self.test_path = task.test_path
        self.data_iterator = None # need to be specified in task

        self.data_pool = {}
        self.model_pool = {}

        self.data_tool_count = defaultdict(int)

        self.best_acc = 0.
        self.best_combo = ()
        self.search_history = {}
        self.model_dev_acc = {}


        self.logger = Logger(f'{self.exp_dir}/log')

        # get gold sents for convenience
        self.dev_gold = self.task.get_tgt_from_file(self.dev_path)



    def new_exp(self):
        # os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f'{self.exp_dir}/data', exist_ok=True)
        os.mkdir(f'{self.exp_dir}/models')
        for tool in self.cfg['params'].keys():
            os.mkdir(f'{self.exp_dir}/models/{tool}')
        os.mkdir(f'{self.exp_dir}/committee')
        os.mkdir(f'{self.exp_dir}/log')
        os.mkdir(f'{self.exp_dir}/tmp')
        # copy original train file to exp/data
        data_path = f'{self.exp_dir}/data/d0.txt'
        self.data_pool['d0'] = data_path # d0 is original data
        copyfile(self.train_path, data_path)


    def resume_exp(self):

        # restore train data
        all_src = []
        for data_path in glob.glob(f'{self.exp_dir}/data/*.txt'):
            data_name = data_path.split('/')[-1].replace('.txt', '')
            print('resume', data_name, data_path)
            self.data_pool[data_name] = data_path
            all_src += self.task.get_src_from_file(data_path)

        # restore data_tool_count
        for model_name in self.model_pool:
            data, tool = model_name.split('.')[1:3]
            self.data_tool_count[(data, tool)] += 1

        # restore data iterator
        self.data_iterator.selected = set(all_src)

        # restore models
        try:
            for line in open(f'{self.exp_dir}/log/model_dev_acc.log'):
                items = line.split('\t')
                model_name = items[0]
                acc = float(items[1])
                self.model_dev_acc[model_name] = acc

            # keep only models with acc
            for model_path in glob.glob(f'{self.exp_dir}/models/*/*.mdl'):
                model_name = model_path.split('/')[-1].replace('.mdl', '')
                if model_name in self.model_dev_acc:
                    self.model_pool[model_name] = model_path
        except:
            print('WARNING: model_dev_acc.log not found')

        # print('model_pool', self.model_pool)
        print('model_dev_acc', self.model_dev_acc)




    def train_func(self, tool, model_name, model_path, train_path, param):
        '''
        call training script to train a model
        then evaluate on dev set and return the accuracy
        '''

        self.logger.log('models', f'start training {model_path}')
        args = [f'{self.script_dir}/train_{tool}.sh', model_path, train_path, self.dev_path, param]

        # handle and log training exceptions
        try:
            s = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        except:
            self.logger.log('error', 'train error: ' + ' '.join(args))
            exit(1)


    def train_all(self, train_data):
        """
        train one batch of instance of each tool in parallel
        """

        self.logger.log('general', f'train all tools on {train_data}')
        processes = []
        for tool in self.cfg['params']:
            for param in self.cfg['params'][tool]:
                train_path = self.data_pool[train_data]
                model_name = f'{train_data}.{tool}.{param}'

                model_path = '%s/models/%s/%s.mdl' % (self.exp_dir, tool, model_name)
                self.data_tool_count[(train_data, tool)] += 1
                self.model_pool[model_name] = model_path

                p = mp.Process(target=self.train_func, args=(tool, model_name, model_path, train_path, str(param)))
                p.start()
                processes.append(p)

        # wait until all training finished
        for p in processes:
            p.join()

        # then log the model accuracies
        for tool in self.cfg['params']:
            for param in self.cfg['params'][tool]:
                model_name = f'{train_data}.{tool}.{param}'
                acc = self.test_model(model_name)
                self.logger.log('models', f'done training {model_name}, dev acc = {acc:.2f}')


        self.logger.log('general', 'done training')

    def predict(self, tool, model_path, input_path, output_path):
        tool = model_path.split('/')[-1].split('.')[1]
        # print('predicting', tool)
        args = [f'{self.script_dir}/pred_{tool}.sh', model_path, input_path, output_path]
        try:
            s = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        except:
            self.logger.log('error', 'pred error: ' + ' '.join(args))
            exit(1)

    def get_tgt_from_model(self, model_name):
        output_path = f'{self.exp_dir}/tmp/dev.{model_name}.txt' 

        # check first whether the output is already there, if not then do the prediction first
        if not os.access(output_path, os.R_OK):
            tool = model_name.split('.')[1]
            model_path = f'{self.exp_dir}/models/{tool}/{model_name}.mdl'
            self.predict(tool, model_path, self.dev_path, output_path)
        return self.task.get_tgt_from_file(output_path)

    def test_model(self, model_name):
        '''
        test model on dev, and log it
        '''
        if model_name not in self.model_dev_acc:
            pred = self.get_tgt_from_model(model_name)
            acc = utils.evaluate(self.dev_gold, pred)
            # keep track of acc of each model, in a dictionary and in a log file for resuming experiment
            self.model_dev_acc[model_name] = acc
            with open(f'{self.exp_dir}/log/model_dev_acc.log', 'a') as out:
                out.write(f'{model_name}\t{acc:.2f}\n')
        else:
            acc = self.model_dev_acc[model_name]
        # print('<test_model>', model_name, acc)
        # print(self.model_dev_acc)
        return acc

    def test_combo(self, combo, mode = 'test'):
        assert mode in ['dev', 'test']

        if mode == 'test':
            input_path = self.test_path
            gold_data = self.test_gold
        else:
            input_path = self.dev_path
            gold_data = self.dev_gold

        preds = [self.get_tgt_from_model(model_name) for model_name in combo]
        voted = utils.vote(preds)
        acc = utils.evaluate(gold_data, voted)
        return acc



############################################
# SEARCH RELATED FUNCTIONS
    def latest_model_as_combo(self):
        best_combo = (list(self.model_pool.keys())[-1],)
        dev_acc = self.test_combo(best_combo, 'dev')
        improve = (dev_acc > self.best_acc)
        if improve:
            self.best_acc = dev_acc
            self.best_combo = best_combo
        return dev_acc, best_combo, improve


    def genetic_search_combo(self):
        self.logger.log('general', 'genetic search combo')

        best_acc, best_combo = 0, ()

        sorted_model_pool =  sorted(self.model_pool)
        preds = [self.get_tgt_from_model(model) for model in sorted_model_pool] # sorted by model name
        model2idx = {c:i for i,c in enumerate(sorted_model_pool)}

        merged_preds = utils.merge(self.dev_gold, preds, transpose=True)

        def eval_func(mask):
            idx = tuple(i for i, x in enumerate(mask) if x)
            if not idx:
                return 0
            else:
                return utils.vote_and_eval(merged_preds, idx)

        best_mask, best_acc = utils.genetic_search(eval_func, len(self.model_pool), 100,
                                                    self.cfg['num_search'], self.cfg['max_combo_size'])
                                                    # min(self.num_search, 2 ** len(self.model_pool)), self.max_combo_size)
        best_combo = tuple(sorted_model_pool[i] for i,x in enumerate(best_mask) if x)


        # log dev and test acc
        # replace best_combo if needed
        dev_acc = self.test_combo(best_combo, 'dev')
        test_acc = self.test_combo(best_combo, 'dev')
        self.logger.log('test', 'dev acc = %.2f, test acc = %.2f' % (dev_acc, test_acc))
        # self.logger.log('general', 'dev acc = %.2f, test acc = %.2f' % (dev_acc, test_acc))

        msg = 'best_acc = %.2f num_models = %d' % (dev_acc, len(best_combo))
        # self.logger.log('general', msg)
        self.logger.log('combo', msg + '\n' + '\n'.join([m for m in best_combo]))

        improve = (dev_acc > self.best_acc)
        if improve:
            self.best_acc = dev_acc
            self.best_combo = best_combo

        return dev_acc, best_combo, improve


    def select_new_data(self):
        self.logger.log('general', 'vote and select')
        preds = []
        
        # write next batch of extra data into a file
        input_path = f'{self.exp_dir}/tmp/extra_data.txt'
        self.data_iterator.get_next(input_path, self.cfg['num_input_data'])
        src = self.task.get_src_from_file(input_path)

        for model_name in self.best_combo:
            tool = model_name.split('.')[2]
            model_path = self.model_pool[model_name]

            output_path = f'{self.exp_dir}/tmp/dev.{model_name}.txt'
            self.predict(tool, model_path, input_path, output_path)

            tgt = self.task.get_tgt_from_file(output_path)
            preds.append(tgt)

        n = len(self.data_pool)
        self.data_pool[f'd{n}'] = f'{self.exp_dir}/data/d{n}.txt'

        # get agreement rate for each sentence 

        # for simiulate single model self training
        if self.cfg.get('self_training', False):
            model_name = list(self.model_pool.keys())[-1]
            output_path = f'{self.exp_dir}/tmp/dev.{model_name}.txt'
            agreements = self.task.get_confidence_from_file(output_path)
        else:
            agreements = []
            for inst_preds in zip(*preds):
                agree = utils.get_agreement(inst_preds)
                # agree = utils.ngram_agreement(sent_preds, 3).mean() # agreement ratio for the sentence
                agreements.append(agree)

        # majority vote for each sentence (a bit redundant, but fine)
        voted = utils.vote(preds)

        print('src', len(src))
        print('voted', len(voted))
        print('preds', len(preds))
        print('agreement', len(agreements))

        # select the instances with high agreement and add as new training data on top of d0
        copyfile(self.data_pool['d0'], self.data_pool[f'd{n}'])

        # ALTERNATIVE: 
        # select the instances with high agreement and add as new training data on top of the previous data version
        # copyfile(self.data_pool[f'd{n-1}'], self.data_pool[f'd{n}'])


        num_selected = 0
        with open(self.data_pool[f'd{n}'], 'a') as out:
            for agree, s, t in sorted(zip(agreements, src, voted), reverse=True)[:self.cfg['num_output_data']]:
                if agree > self.cfg['min_agreement']:
                    num_selected += 1
                    out.write(f'{s}\t{t}\n')
                    # remember in extra data
                    self.data_iterator.selected.add(s)

        self.logger.log('general', f"selected {num_selected} / {self.cfg['num_input_data']} instances")

        return f'd{n}'



############################################
# START

    def start(self):
        if os.access(f'{self.exp_dir}/log', os.R_OK):
            print(f'Resume Experiment {self.exp_dir}')
            self.resume_exp()
        else:
            print(f'New Experiment {self.exp_dir}')
            self.new_exp()

        # new experiment
        if len(self.model_dev_acc) == 0:
            train_data = 'd0'
            new_models = self.train_all(train_data)

        if self.cfg.get('self_training', False):
            dev_acc, combo, improve = self.latest_model_as_combo()
        else:
            dev_acc, combo, improve = self.genetic_search_combo()

        no_improve = 0
        while no_improve < 10:
 
            # (1) select extra data
            if self.cfg.get('no_extra', False):
                print('no extra data')
                n = len(self.data_pool)
                self.data_pool[f'd{n}'] = self.data_pool['d0']
                train_data = f'd{n}'
            else:
                train_data = self.select_new_data()

            # (2) train new models
            new_models = self.train_all(train_data)

            # (3) find the best model combination given the new models
            if self.cfg.get('self_training', False):
                dev_acc, combo, improve = self.latest_model_as_combo()
            else:
                dev_acc, combo, improve = self.genetic_search_combo()
            if not improve:
                no_improve += 1

            # (4) maintain the size of the pool
            # print('current_dev_acc', list(self.model_dev_acc.keys()))
            print('current_pool:', len(self.model_pool))
            best_models = sorted(self.model_dev_acc.keys(), key=lambda x: -self.model_dev_acc[x])[:self.cfg['max_model_pool_size']]
            # print('best_models', best_models)
            print('current_combo:', combo)
            print('best_combo:', self.best_combo)
            for model in list(self.model_pool.keys()):
                if model not in combo and model not in self.best_combo and model not in best_models:
                    print(f'remove {model}')
                    model_file = self.model_pool.pop(model)
                    files = glob.glob(f'{model_file}*')
                    for f in files:
                        os.remove(f)
            # print('after_dev_acc', list(self.model_dev_acc.keys()))
            print('after_pool:', len(self.model_pool))


if __name__ == '__main__':
    exp = Experiment(*sys.argv[1:])
    exp.start()