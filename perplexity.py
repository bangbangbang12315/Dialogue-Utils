import os
import requests
import io 
import pkuseg
import json
import itertools
import logging
import argparse
from tqdm import tqdm 
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk import word_tokenize, sent_tokenize 
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk import bigrams, FreqDist
from itertools import chain

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):

    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):

    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    # return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)
    sentencelist = []
    for s in sentences:
        sentencelist.extend(s)
    return distinct_n_sentence_level(sentencelist,n)
def ppl(textTest,train,n_gram=4):
    n = n_gram
    tokenized_text = [list(map(str.lower, sent)) 
                    for sent in train]
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

    tokenized_text = [list(map(str.lower, sent)) 
                    for sent in textTest]
    test_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

    model = Laplace(1) 
    model.fit(train_data, padded_sents)

    s = 0
    for i, test in enumerate(test_data):
        p = model.perplexity(test)
        s += p
    return s / (i + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',default="resp.test", help="test file",type=str)
    parser.add_argument('--infer',default="moe_legal_1.txt", help="inference file",type=str)
    args = parser.parse_args()
    if not os.path.exists('logging'):
        os.mkdir('logging')
    logging_fp = 'logging/' + args.infer + '.log'
    log_formatter = logging.Formatter('%(message)s')
    log_handler = logging.FileHandler(logging_fp)
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger('eval')
    logger.addHandler(log_handler)
    logger.setLevel(level=logging.INFO)
    corpus_train = args.test
    text = []
    num = 0
    idx = 0
    out_file = open(args.infer,'r')
    candidate = []
    bleu_score_all_1 = 0
    bleu_score_all_2 = 0
    bleu_score_all_3 = 0
    bleu_score_all_4 = 0
    train_sentence = []
    for line in out_file:
        r = line.split(' ')
        candidate.append(r)
    with open(corpus_train,'r') as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            reference = []
            data = line
            resps_num = len(data)
            for resp in data.split('\t'):
                reference.append(resp.split(' '))
                train_sentence.append(resp.split(' '))
            bleu_score_1 = sentence_bleu(reference, candidate[idx],weights=(1, 0, 0, 0))
            bleu_score_all_1 += bleu_score_1
            bleu_score_2 = sentence_bleu(reference, candidate[idx],weights=(0, 1, 0, 0))
            bleu_score_all_2 += bleu_score_2
            bleu_score_3 = sentence_bleu(reference, candidate[idx],weights=(0, 0, 1, 0))
            bleu_score_all_3 += bleu_score_3
            bleu_score_4 = sentence_bleu(reference, candidate[idx],weights=(0, 0, 0, 1))
            bleu_score_all_4 += bleu_score_4
            num += 1
    # ppl_score_1 = ppl(candidate,train_sentence,1)
    # ppl_score_2 = ppl(candidate,train_sentence,2)
    distinct_score_1 = distinct_n_corpus_level(candidate,1)
    distinct_score_2 = distinct_n_corpus_level(candidate,2)
    logger.info('BLEU-1:%f, BLEU-2:%f,BLEU-3:%f,BLEU-4:%f,DISTINCT-1:%f,DISTINCT-2:%f',
        bleu_score_all_1 / num, bleu_score_all_2 / num, bleu_score_all_3 / num, bleu_score_all_4 / num,
        distinct_score_1,distinct_score_2)
    out_file.close()
