from tqdm import tqdm
import os

dir_path = '../data/Pdata/reddit'
phase = ['train', 'dev', 'test']
filelist = ['post', 'resp']
output_path = os.path.join(dir_path, 'sentences.txt')
with open(output_path, 'w') as fout:
    for p in phase:
        for fp in filelist:
            f = open(os.path.join(dir_path, p, fp+'.'+p), 'r')
            for line in tqdm(f, desc='Load {}'.format(fp+'.'+p)):
                if 'weibo' in dir_path:
                    line = ''.join(line.strip().split(' '))
                else:
                    line = line.strip()
                fout.write(line+'\n')

