import json_lines
import sys

def convert(jsonl_file, src_file, tgt_file):
    with open(jsonl_file, 'rb') as f,\
         open(src_file, 'w') as src,\
         open(tgt_file, 'w') as tgt: 
        for item in json_lines.reader(f): 
            string = item['string'].replace('\n', '')
            src.write(f"{string}\n")
            tgt.write(f"{item['label']}\n")

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
