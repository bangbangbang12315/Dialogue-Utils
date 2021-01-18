import json
import re
from tqdm import tqdm
from collections import Counter, defaultdict

from statistics import Statistics


def load(fp, patition=None, num=500000):
    def clean(subs):
        if re.search(r'【.*】', subs[0]): return False
        if len(subs[0].split(' ')) > 60: return False
        return True
    data = []
    cnt = 0
    with open(fp, 'r') as f:
        for line in tqdm(f, desc="Load Data: "):
            # if cnt >= 500000:   #拆分小数据集
            #     break
            cnt += 1
            if not cnt % 2:
                subs += line.strip().split('\t')
                assert len(subs) == 8
                if not patition: data.append(subs)
                elif subs[6] == patition or subs[6] in patition: 
                    if clean(subs):
                        data.append(subs)
                else: break
            else:
                subs = line.strip().split('\t')
    return data

def filter_data(data):
    # data = [subs for subs in data if not re.search(r'【.*】', subs[0])]
    # data = [subs for subs in data if len(subs[0].split(' ')) <= 60]
    history_cnt = Counter([subs[4] for subs in data])
    data = [subs for subs in data if history_cnt[subs[4]] >= 1000]
    # after filter
    history_cnt = Counter([subs[4] for subs in data])
    Statistics.get_bucket(history_cnt.values())
    return data

def train_dev_test(data_dir, data):
    cnt = 0
    def get_history(uid, ts):
        history = []
        ts = int(ts)
        for his in user_history[uid]:
            if int(his[5]) < ts:
                history.append(his)
            else: break
        return json.dumps(history, ensure_ascii=False)

    user_history = defaultdict(list)
    # p p_id p_time r r_id r_time part train/dev/test
    for subs in data:
        if subs[-1] == '0': user_history[subs[4]].append(subs[:6])
    for uid, his in tqdm(user_history.items(), desc="Sort Hisotry: "):
        user_history[uid] = sorted(his, key=lambda x: int(x[5]))

    dataset_objs = [open(f"{data_dir}/{phase}.raw", 'w') 
                        for phase in ['train', 'dev', 'test']]
    for subs in tqdm(data, desc="Train/Dev/Test: "):
        cnt += 1
        p, p_uid, p_time, r, r_uid, r_time, _, phase = subs
        his = get_history(r_uid, r_time)
        new_subs = [p, p_uid, p_time, r, r_uid, r_time, his]
        if cnt % 10 == 0:
            dataset_objs[1].write(json.dumps(new_subs, ensure_ascii=False) + '\n')
        else:
            dataset_objs[int(phase)].write(json.dumps(new_subs, ensure_ascii=False) + '\n')
    
    for obj in dataset_objs:
        obj.close()

def prepare_model_data(model_data_dir):
    sub_files = ['post', 'post_uid', 'post_time', 'resp', 'resp_uid', 'resp_time', 'history_post', 'history_resp', 'history_extra']
    for phase in ['train', 'dev', 'test']:
        raw_fp = f"{model_data_dir}/raw/{phase}.raw"
        sub_file_objs = [open(f"{model_data_dir}/{phase}/{sub_file}.{phase}", 'w') for sub_file in sub_files]
        with open(raw_fp, 'r') as f:
            for line in tqdm(f, desc=f"Model {phase} data: "):
                subs = json.loads(line)
                # print([len(subs), len(sub_file_objs)])
                assert len(subs) == len(sub_file_objs)
                # [p, p_uid, p_time, r, r_uid, r_time, his_post, his_resp, his_extra]
                for obj, info in zip(sub_file_objs, subs):
                    obj.write(info + '\n')
        for obj in sub_file_objs:
            obj.close()


def get_similarity_corpus(fp, data):
    history = defaultdict(set)
    with open(fp, 'w') as f:
        for subs in data:
            p, p_id, p_time, r = subs[:4]
            key = hash(p + p_id)
            if p_time not in history[key]:
                f.write(f"{p}\n")
                history[key].add(p_time)
            f.write(f"{r}\n")

def get_user_data(model_data_dir, data):
    '''
    return {user_1: sentence_1, sentence_2, ...}
    '''
    user_data = defaultdict(list)
    for subs in tqdm(data, desc="get_user_data"):
        user_data[subs[4]].append(subs[3])
    with open(model_data_dir, 'w') as fdata:
        fdata.write(json.dumps(user_data, ensure_ascii=False))
        
if __name__ == "__main__":
    data_fp = "../../data/PchatbotW.release_ver"
    model_data_dir = "../data/user_test"
    data = load(data_fp, patition=['1'])
    data = filter_data(data)
    # get_similarity_corpus(similarity_corpus_fp, data)
    get_user_data(model_data_dir + "/user_data", data)
    # prepare_model_data(model_data_dir)

    

    