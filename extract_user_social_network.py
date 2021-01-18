import json
import re
from tqdm import tqdm
import os
from collections import Counter, defaultdict

from statistics import Statistics
uf = defaultdict(int)
cnt = defaultdict(int)
com = defaultdict(list)

total = 0
def find(p):
    while p != uf[p]:
        uf[p] = uf[uf[p]]
        p = uf[p]
    return p

def union(p,q):
    global total
    rootp = find(p)
    rootq = find(q)
    if rootp == rootq:
        return
    else:
        if cnt[rootp] > cnt[rootq]:
            uf[rootq] = rootp
            cnt[rootp] += cnt[rootq]
        else:
            uf[rootp] = rootq
            cnt[rootq] += cnt[rootp]
        total -= 1

def load(in_path):
    global total
    filenames = sorted(os.listdir(in_path))
    puser = defaultdict(list)
    for filename in tqdm(filenames):
        fhand = open(os.path.join(in_path,filename))
        data = fhand.readlines()
        data = json.loads(data[0])
        for sub in data:
            p, pid, ptime, r, rid, rtime = sub
            if cnt[pid] == 0:
                total += 1
            if cnt[rid] == 0:
                total += 1
            puser[rid].append(pid)
            uf[pid] = pid
            uf[rid] = rid
            cnt[rid] = 1
            cnt[pid] = 1
    print("Total user: {}".format(str(total)))
    return puser

def prepross(puser, threhold=2, hop=5):
    for rid, v in tqdm(puser.items()):
        for pid in v:
            union(rid, pid)
    for rid in tqdm(puser.keys()):
        com[uf[rid]].append(rid)
    store_data = '../data/socail_network/'
    for k, v in com.items():
        with open(store_data+str(k), 'w') as fuser:
            s = '\n'.join(v)
            fuser.write(s)
    print("Total Comunity: {}".format(str(total)))

def store_data(data, model_data_dir, phase):
    post, resp, user =[], [], []
    with open(os.path.join(model_data_dir, phase, 'post.'+phase), 'w') as fpost, open(os.path.join(model_data_dir, phase, 'resp.'+phase), 'w') as fresp, open(os.path.join(model_data_dir, phase, 'resp_id.'+phase), 'w') as fuser:
        for line in data:
            p, p_uid, p_time, r, r_uid, r_time, _, phase = line
            fpost.write(p+'\n')
            fresp.write(r+'\n')
            fuser.write(r_uid+'\n')

def extract(data_dir, store_dir, seed_user, threhold=2, hop=1):
    puser = defaultdict(int)
    with open(data_dir+seed_user, 'r') as fuser:
        data = fuser.readlines()
        data = json.loads(data[0])
    for sub in tqdm(data):
        p, pid, ptime, r, rid, rtime = sub
        puser[pid] += 1
    with open(store_data+seed_user, 'w') as fuser:
        for k, v in puser.items():
            if v >= threhold:
                fuser.write(k+' '+str(v)+'\n')

if __name__ == "__main__":
    data_dir = '../data/weibo_user/'
    store_data = '../data/socail_network/'
    seed_user = '174070'
    # extract(data_dir, store_data, seed_user)
    puser = load(data_dir)
    prepross(puser)

