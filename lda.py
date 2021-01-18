# 使用gensim中的潜在狄利克雷分配对题目文本进行主题分析
from LAC import LAC
import gensim
from gensim import corpora, models
import pandas as pd
import json
import re
from tqdm import tqdm
import os
from collections import Counter, defaultdict
from statistics import Statistics

def load_stop_words(stop_dir):
    stop_words = set()
    with open(stop_dir, 'r') as fp:
        for line in fp:
            stop_words.add(line.strip())
    return stop_words

def load(in_path):
    filenames  =  sorted(os.listdir(in_path))
    user_data = defaultdict(list)
    cnt = 0
    user = 0
    for filename in tqdm(filenames):
        fhand = open(os.path.join(in_path,filename))
        user += 1
        # if user > 70000:
        #     break
        data = fhand.readlines()
        data = json.loads(data[0])
        for sub in data:
            p, pid, ptime, r, rid, rtime = sub
            user_data[rid].append(r)
        fhand.close()
    return user_data

def prepross(user_data, stop_words, model_dir, num_topics=10):
    texts = []
    for user, doclist in tqdm(user_data.items()):
        text = []
        for doc in doclist:
            for word in doc.split(' '):
                if word not in stop_words:
                    text.append(word)
        texts.append(text)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)  # num_topics为主题数
    lda.save(model_dir)

def store_data(data, model_data_dir, phase):
    post, resp, user =[], [], []
    with open(os.path.join(model_data_dir, phase, 'post.' + phase), 'w') as fpost, open(os.path.join(model_data_dir, phase, 'resp.' + phase), 'w') as fresp, open(os.path.join(model_data_dir, phase, 'resp_id.' + phase), 'w') as fuser:
        for line in data:
            p, p_uid, p_time, r, r_uid, r_time, _, phase = line
            fpost.write(p+'\n')
            fresp.write(r+'\n')
            fuser.write(r_uid+'\n')

def infer(model_dir):
    lda = gensim.models.ldamodel.LdaModel.load(model_dir)
    print(lda.print_topics(num_topics=10, num_words=20))


if __name__ == "__main__":
    data_fp = '../data/legal_user'
    model_dir = 'legal.lda.model'
    stop_dir = './Dialog-Preprocessor/preprocess/stopwords.txt'
    infer(model_dir)
    # data = load(data_fp)
    # stop_words = load_stop_words(stop_dir)
    # prepross(data, stop_words, model_dir, 10)
    
