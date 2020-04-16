import json_lines
import sys

def convert(train_jsonl, dev_jsonl, test_jsonl, txt_file, tgt_file):
    data = []
    tgts = []
    train_idx = []
    dev_idx = []
    test_idx = []

    idx = 0
    with open(train_jsonl, 'rb') as f:
        for item in json_lines.reader(f): 
            data.append(item['string'].lower().replace('\n', ''))
            tgts.append(item['label'])
            train_idx.append(idx)
            idx += 1
    with open(dev_jsonl, 'rb') as f:
        for item in json_lines.reader(f): 
            data.append(item['string'].lower().replace('\n', ''))
            tgts.append(item['label'])
            dev_idx.append(idx)
            idx += 1
    with open(test_jsonl, 'rb') as f:
        for item in json_lines.reader(f): 
            data.append(item['string'].lower().replace('\n', ''))
            tgts.append(item['label'])
            test_idx.append(idx)
            idx += 1

    with open(txt_file, 'w') as f:
        for s in data:
            f.write(f'{s}\n')
    # with open(tgt_file, 'w') as f:
    #     for t in tgts:
    #         f.write(f'{t}\n')

    with open(train_jsonl.replace('jsonl', 'idx'), 'w') as f:
        for i in train_idx:
            f.write(f'{i}\t{tgts[i]}\n')
    with open(dev_jsonl.replace('jsonl', 'idx'), 'w') as f:
        for i in dev_idx:
            f.write(f'{i}\t{tgts[i]}\n')
    with open(test_jsonl.replace('jsonl', 'idx'), 'w') as f:
        for i in test_idx:
            f.write(f'{i}\t{tgts[i]}\n')


if __name__ == '__main__':
    convert(*sys.argv[1:])
