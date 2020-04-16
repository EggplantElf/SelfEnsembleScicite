"""
A simple sentence classfier with MLP on the BERT encoded sentences.
It takes the encoded numpy file as data, and train/dev/test splits 
only contain the sentence ID and label of the main numpy file.

Usage:
    - train:
        python classifier.py train -m [model_file] -s [all_data_numpy_file] -t [train_index_file] -d [dev_index_file]

    - pred:
        python classifier.py pred -m [model_file] -s [all_data_numpy_file] -i [input_index_file] -o [output_index_label_file]

"""

import dynet as dy
import numpy as np
import sys
import random
import tqdm
from argparse import ArgumentParser


def iterate(data):
    while True:
        yield from data
        random.shuffle(data)

class Instance:
    def __init__(self, idx, src, tgt):
        self.idx = idx
        self.src = src
        self.tgt = tgt

def read_data(src_file, id_file):
    data = []
    srcs = list(np.load(src_file).values())
    ids = [int(line.split()[0]) for line in open(id_file)]
    for line in open(id_file):
        items = line.strip().split('\t')
        if len(items) == 2: 
            i, t = int(items[0]), items[1]
        else:
            i, t = int(items[0]), '<???>'
        data.append(Instance(i, srcs[i], t))
    return data

class Classifier:
    def __init__(self, args):
        self.args = args

        # fix for the scicite dataset
        self.labels = ['<???>', 'background', 'method', 'result']
        self.label_map = {l: i for i, l in enumerate(self.labels)}

        if self.args.mode == 'train':
            self.train_set = read_data(self.args.src_file, self.args.train_file)
            self.dev_set = read_data(self.args.src_file, self.args.dev_file)
        else:
            self.test_set = read_data(self.args.src_file, self.args.input_file)

        self.model = dy.Model()
        self.w1 = self.model.add_parameters((128, 768))
        self.b1 = self.model.add_parameters(128)
        self.w2 = self.model.add_parameters((len(self.labels), 128))
        self.b2 = self.model.add_parameters(len(self.labels))

        self.trainer = dy.AdamTrainer(self.model)
        # self.trainer = dy.MomentumSGDTrainer(self.model)

        if self.args.mode != 'train':
            self.load_model()


    def log(self, msg):
        if self.args.mode == 'train' and self.args.model_file:
            with open(self.args.model_file+'.log', 'a') as f:
                f.write(str(msg)+'\n')
        print(msg)

    def save_model(self):
        self.log('saving model')
        self.model.save(self.args.model_file)


    def load_model(self):
        self.log('loading model')
        self.model.populate(self.args.model_file)

    def decode(self, inst, train_mode=False):
        loss = 0

        x = dy.mean_dim(dy.inputTensor(inst.src), [0], False) # (784)
        y = self.b2 + self.w2 * dy.rectify(self.b1 + self.w1 * x)
        sm = dy.softmax(y)
        pidx = sm.npvalue().argmax()
        conf = sm.npvalue().max()

        if train_mode:
            gidx = self.label_map[inst.tgt]
            loss += dy.pickneglogsoftmax(y, gidx)

        return {'pred': self.labels[int(pidx)],
                'conf': conf,
                'loss': loss,
                'correct': int(pidx == gidx)}



    def train(self):
        correct = total = lvalue = 0
        waited = best = 0
        # for i, inst in tqdm(enumerate(iterate(self.train_set), 1)):
        for i, inst in enumerate(iterate(self.train_set), 1):
            dy.renew_cg()
            res = self.decode(inst, True)
            loss = res['loss']
            lvalue += res['loss'].value() if loss else 0
            total += 1
            correct += res['correct']
            if loss:
                loss.backward()
                self.trainer.update()
            if i % self.args.eval_every == 0:
                self.log(f'loss = {lvalue/self.args.eval_every:.2f}, train_acc = {correct} / {total} = {100*correct / total:.2f}%')
                # res = self.predict(self.dev_set[:10])
                res = self.predict(self.dev_set)
                acc = 100*res['correct'] / res['total']
                self.log(f"dev_acc = {res['correct']} / {res['total']} = {acc:.2f}%")
                correct = total = lvalue = 0

                if acc > best:
                    best = acc
                    waited = 0
                    self.save_model()
                else:
                    waited += 1
                    if waited > 5:
                        self.log('Finish training')
                        break

        self.load_model()
        self.log('Final dev')
        res = self.predict(self.dev_set)
        acc = 100*res['correct'] / res['total']
        self.log(f"final_dev_acc = {res['correct']} / {res['total']} = {acc:.2f}%")



    def predict(self, data):
        preds = []
        confs = []
        correct = total = 0

        for inst in data:
            dy.renew_cg()
            res = self.decode(inst, True)
            preds.append(res['pred'])
            confs.append(res['conf'])
            total += 1
            correct += int(res['pred'] == inst.tgt)

        if self.args.output_file:
            with open(self.args.output_file, 'w') as out:
                for inst, pred, conf in zip(data, preds, confs):
                    out.write(f'{inst.idx}\t{pred}\t{conf}\n')
        return {'total': total,
                'correct': correct}




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("mode", choices=['train', 'pred', 'eval'])
    parser.add_argument("-m", "--model_file")
    parser.add_argument("-s", "--src_file")
    parser.add_argument("-t", "--train_file")
    parser.add_argument("-d", "--dev_file")
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-o", "--output_file")
    # parser.add_argument("--confidence", type=float, default=0)
    parser.add_argument("--hid_dim", type=int, default=768)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=20000)

    args = parser.parse_args()
    if args.mode == 'train':
        model = Classifier(args)
        model.train()
    elif args.mode == 'pred':
        model = Classifier(args)
        res = model.predict(model.test_set)
        acc = 100*res['correct'] / res['total']
        print(f"test_acc = {res['correct']} / {res['total']} = {acc:.2f}%")
