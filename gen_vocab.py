# raw_data : weibo_corpus.cutted format data
# data: post.train format data
import sys
import json
import random
import os
from collections import Counter
from tqdm import tqdm

def load_raw_data(fp):
    raw_data = []
    with open(fp, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line.isspace():
                raw_data.append(line)
    return raw_data

def get_data_vocab(data, vocab_num=40000):
    # <unk> <s> </s> first
    # format data/src_vocab_file
    vocab_dic = {}
    for text in data:
        words = text.split(' ')
        for w in words:
            if not w: continue
            vocab_dic[w] = vocab_dic.get(w, 0) + 1
    vocab = [k for k, _ in sorted(vocab_dic.items(), key=lambda x: -x[1])]
    vocab = ['<unk>', '<s>', '</s>'] + vocab[:vocab_num - 3]
    return vocab

def save_data(fp, data):
    print(f"Save {fp}")
    with open(fp, 'w') as f:
        for text in tqdm(data):
            f.write(text + '\n')

if __name__ == "__main__":
    raw_fp, save_dir, vocab_size = sys.argv[1:]
    save_dir = save_dir.rstrip('/')
    vocab_size = int(vocab_size)
    raw_data = load_raw_data(raw_fp)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    vocab = get_data_vocab(raw_data, vocab_num=vocab_size)
    save_data(f"{save_dir}/vocab_file", vocab)