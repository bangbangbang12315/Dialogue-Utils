import json
import re
from tqdm import tqdm
import os
import copy
from collections import Counter, defaultdict
from statistics import Statistics
import random

def gen_history(user_data):
    max_memory = 15
    if len(user_data) == 0:
        return []
    else:
        if len(user_data) > max_memory:
            choose_memory = random.sample(user_data, max_memory)
            return choose_memory
        else:
            return user_data

def load(in_path):
    filenames  =  sorted(os.listdir(in_path))
    def clean(subs):
        if re.search(r'【.*】', subs[0]): return False
        if len(subs[0].split(' ')) > 50 or len(subs[3].split(' ')) > 50: return False
        return True
    train_data = []
    dev_data = []
    test_data = []
    train_his_data = []
    dev_his_data = []
    test_his_data = []
    cnt = 0
    user = 0
    for filename in tqdm(filenames):
        fhand = open(os.path.join(in_path,filename))
        user += 1
        user_data = []
        user_history = []
        if user > 300000:
            break
        for line in fhand:
            cnt += 1
            subs = line.strip().split('\t')
            if len(subs) != 8:
                continue
            if clean(subs):
                his = gen_history(copy.deepcopy(user_data))
                user_history.append(his)
                user_data.append(subs)
        if len(user_data) < 4:
            continue
        train_data.extend(user_data[:-4])
        train_his_data.extend(user_history[:-4])
        dev_data.extend(user_data[-4:-2])
        dev_his_data.extend(user_history[-4:-2])
        test_data.extend(user_data[-2:])
        test_his_data.extend(user_history[-2:])
        fhand.close()
    return train_data, dev_data, test_data, train_his_data, dev_his_data, test_his_data

def store_data(data, model_data_dir, phase):
    post, resp, user =[], [], []
    label = ''

    if not os.path.exists(os.path.join(model_data_dir, phase.split('_')[0])):
        os.mkdir(os.path.join(model_data_dir, phase))
    if 'his' in phase:
        phase, label = phase.split('_')
    else:
        label = phase
        fuser = open(os.path.join(model_data_dir, phase, 'resp_id.'+phase), 'w')

    with open(os.path.join(model_data_dir, phase, 'post.'+label), 'w') as fpost, open(os.path.join(model_data_dir, phase, 'resp.'+label), 'w') as fresp:
        for line in data:
            if len(line) == 0:
                p, r, r_uid = '<\s>', '<\s>', '<\s>'
            elif label != 'his':
                p, p_uid, p_time, r, r_uid, r_time, _, phase = line
                fuser.write(r_uid+'\n')
            else:
                p, r = '', '' 
                for his in line:
                    p_his, p_uid, p_time, r_his, r_uid, r_time, _, phase = his
                    if len(p) == 0:
                        p, r = p_his, r_his
                    else:
                        p += '\t' + p_his
                        r += '\t' + r_his
            fpost.write(p+'\n')
            fresp.write(r+'\n')
                
if __name__ == "__main__":
    data_fp = "../data/PChatbot_byuser_filter"
    model_data_dir = "../data/Pdata/weibo"
    train_data, dev_data, test_data, train_his_data, dev_his_data, test_his_data = load(data_fp)
    store_data(train_data, model_data_dir, 'train')
    store_data(dev_data, model_data_dir, 'dev')
    store_data(test_data, model_data_dir, 'test')
    store_data(train_his_data, model_data_dir, 'train_his')
    store_data(dev_his_data, model_data_dir, 'dev_his')
    store_data(test_his_data, model_data_dir, 'test_his')

    