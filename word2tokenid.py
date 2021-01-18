import sentencepiece as spm
from tqdm import tqdm
import os
import re

dir_path = '../data/Pdata/reddit'

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(dir_path, 'spm_8000.model'))

unk_token = 0
sos_token = 1
eos_token = 2

def test():
    text = '我想你啦！'
    tokenized = sp.EncodeAsIds(text)
    print(tokenized)
    retext = sp.DecodeIds(tokenized)
    print(retext)
    print(sp.pad_id())

# def prepross(line, target):
#     if 'his' in target:
#         line = line.replace('\t', eos)
#     line = line.replace(' ','')
#     return line

def tokenize(input_path, output_path):
    target = os.path.split(input_path)[-1]
    cnt = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, desc='tokenizing {}'.format(target)):
            cnt += 1
            # if cnt > 100:
            #     break
            line = line.strip()
            if 'his' in target:
                if cnt == 1:
                    tokenized = [2]
                else:
                    tokenized = []
                    for utterance in line.split('\t'):
                        if 'weibo' in dir_path:
                            utterance = utterance.replace(' ','')
                        utterance_ids = sp.EncodeAsIds(utterance) + [2]
                        tokenized.extend(utterance_ids)
            else:
                if 'weibo' in dir_path:
                    line = line.replace(' ', '')
                tokenized = sp.EncodeAsIds(line)
            if target in ['resp.train', 'resp.dev', 'resp.test']:
                tokenized = [sos_token] + tokenized + [eos_token]
            fout.write(' '.join(list(map(lambda x: str(x), tokenized))) + '\n')

if __name__ == "__main__":
    # test()
    phase = ['train', 'dev', 'test']
    filelist = ['post', 'resp']
    for p in phase:
        for f in filelist:
            input_path = os.path.join(dir_path, p, f+'.'+p)
            output_path = os.path.join(dir_path, p, f+'.'+p+'.'+'id'+'.8000')
            tokenize(input_path, output_path)
        for f in filelist:
            input_path = os.path.join(dir_path, p, f+'.'+'his')
            output_path = os.path.join(dir_path, p, f+'.'+'his'+'.'+'id'+'.8000')
            tokenize(input_path, output_path)