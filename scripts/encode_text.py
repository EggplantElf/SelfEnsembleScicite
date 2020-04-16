"""
Encode all sentences just once with BERT and save to a numpy file,
so that later we don't need to repeatedly encode them.

Usage:
    python encode_text.py [one_sent_per_line_text_file] [output_numpy_file] ['bert' or 'scibert']
"""

import sys
import torch
from transformers import BertModel, BertTokenizer
import numpy as np

def encode(input_file, output_file, model_type='scibert'):
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights =  './models/scibert-scivocab-uncased' if model_type == "scibert" else 'bert-base-uncased'
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    data = []

    # Encode text
    for line in open(input_file):
        text = line.strip().lower()
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)[:512]])
        print(text)
        print(input_ids)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
            out = last_hidden_states.numpy().reshape((-1,768))
            print(out.shape)
            data.append(out)

    print('store data')
    np.savez(output_file, *data)

if __name__ == '__main__':
    encode(sys.argv[1], sys.argv[2], sys.argv[3])