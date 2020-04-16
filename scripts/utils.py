"""
Generic helper functions for the ensemble self-learning experiments
"""


import sys
from collections import Counter, defaultdict
import numpy as np
import operator
import random
import math
import time
import heapq as hq

def get_src_from_file(filename):
    data = []
    for line in open(filename):
        data.append(line.strip().split('\t')[0])
    return data

def get_tgt_from_file(filename):
    data = []
    for line in open(filename):
        data.append(line.strip().split('\t')[-1])
    return data


def evaluate(gold, pred):
    total = correct = 0
    for g, p in zip(gold, pred):
        total += 1
        correct += int(g == p)
    return 100 * correct / total

def vote(preds):
    '''
    preds is list of list of predictions (one sentence each)
    '''
    voted = [majority(sent_preds) for sent_preds in zip(*preds)] 
    return voted

def majority(lst):
    return Counter(lst).most_common(1)[0][0]


def vote_and_eval(preds, combo):
    '''
    vote and evluate at the same time
    requires preds encode gold tag with prefix @,
    also the eval score is relative, since the preds have already removed unanimous decisions
    indices are the chosen preds for evaluation
    '''
    total = len(preds)
    correct = 0

    for token_preds in preds:
        tgt = majority([token_preds[i] for i in combo])
        correct += (tgt.startswith('@'))
    return correct * 100. / total 

def merge(gold, preds, transpose = True):
    '''
    merge the gold information into the preds, 
    also remove the unanimous correct preds to make search faster,
    by default, the output is transposed (list of predictions for each token)
    '''
    out = []
    for g, ps in zip(gold, zip(*preds)):
        # mark the correct predictions with prefix @
        bps = tuple(('@'+p if p == g else p) for p in ps) 
        # also remove the vast majority (> 90% on one decision)
        if bps.count(majority(bps)) / len(bps) < 0.9:
            out.append(bps)

    print(f'before: {len(gold)}, after: {len(out)}')

    if transpose:
        return out
    else:
        return zip(*out)

def get_agreement(preds):
    c = Counter(preds).most_common(1)
    return 0 if c[0][0] == '' else c[0][1] / len(preds)



def genetic_search(eval_func, num_models, pool_size, num_iters, max_combo_size=0):
    t0 = time.time()
    history = set() # [combo,...]
    # pool = set() # [(acc, combo),...]
    pool = []
    cross_over_rate = 0.6
    mutation_rate = 0.01
    patience = 100
    trials = 0
    total_trials = 0
    total_individuals = 0

    # init population with random ones
    for i in range(pool_size):
        combo = tuple(random.randint(0, 1) for _ in range(num_models))
        acc = eval_func(combo)
        history.add(combo)
        # pool.add((acc, combo))
        hq.heappush(pool, (acc, combo))

    # selection process
    for i in range(num_iters):

        new_species = False
        total_individuals += 1

        while not new_species:
            total_trials += 1
            # parent selection
            # p1 and p2 are the combo with the maximum acc in the randomly sampled group
            p1 = max(random.sample(pool, pool_size // 10))[1] 
            p2 = max(random.sample(pool, pool_size // 10))[1] 

            # cross over
            child = tuple(g1 if random.random() > cross_over_rate else g2 for g1, g2 in zip(p1, p2))

            # mutation
            mutated = tuple(g if random.random() > mutation_rate else (1 - g) for g in child)

            # limit combo size
            if max_combo_size:
                selected = [i for i, m in enumerate(mutated) if m]
                if len(selected) > max_combo_size:
                    idx = random.sample(selected, max_combo_size)
                    mutated = [0] * num_models
                    for i in idx:
                        mutated[i] = 1
                    mutated = tuple(mutated)


            if mutated in history:
                trials += 1
                if trials > patience:
                    break
            else:
                new_species = True
                acc = eval_func(mutated)
                history.add(mutated)
                # pool.add((acc, mutated))
                # pool.remove(min(pool))
                hq.heappushpop(pool, (acc, mutated))
                trials = 0

        # print i, max(pool, key = lambda x: x[1])

    # return the combo acc pair
    acc, combo = max(pool)
    print(f'time for genetic search: {(time.time() - t0):.1f}')
    print(f'total trials: {total_trials}, total_individuals: {total_individuals}')
    return combo, acc

def random_search(num_models, pool_size, num_iters, eval_func):
    '''
    just as a baseline for genetic search
    '''
    history = set() # [combo,...]
    pool = set() # [(acc, combo),...]
    patience = 100
    trials = 0
    t0 = time.time()

    # selection process
    for i in range(num_iters + 100):

        new_species = False
        while not new_species:
            combo = tuple(random.randint(0, 1) for _ in range(num_models))

            if combo in history:
                trials += 1
                if trials > patience:
                    break
            else:
                new_species = True
                acc = eval_func(combo)
                history.add(combo)
                pool.add((acc, combo))
                if len(pool) > 100:
                    pool.remove(min(pool))
                

        # print i, max(pool, key = lambda x: x[1])

    # return the combo acc pair
    acc, combo = max(pool)
    print(f'time for random search: {(time.time() - t0):.1f}')
    return combo, acc

