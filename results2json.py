from tqdm import tqdm
import json
import pkuseg
def load(fp):
    data = []
    seg = pkuseg.pkuseg()
    with open(fp, 'r') as fsrc:
        for line in tqdm(fsrc, desc='Loading Data'):
            if fp != './evaluate/mmi.txt':
                data.append(line.strip().split(' '))
            else:
                data.append(seg.cut(line.strip()))

    return data
def store(post, answer, history, result, fp):
    with open(fp, 'w') as ftgt:
        for p, a, r in zip(post, answer, result):
            dic = {}
            dic['post'] = p
            dic['answer'] = a
            dic['history'] = []
            dic['result'] = r
            ftgt.write(json.dumps(dic, ensure_ascii=False)+'\n')

if __name__ == "__main__":
    # fpost = '../data/reddit_new_data/test/post.tiny.test'
    # fresp = '../data/reddit_new_data/test/resp.tiny.test'
    # fresult = './evaluate/mmi_reddit2.txt'
    # ftgt = './result/mmi_reddit_json.txt'
    fpost = '../data/baseline_data/test/post.tiny.test'
    fresp = '../data/baseline_data/test/resp.tiny.test'
    fresult = './evaluate/mmi.txt'
    ftgt = './result/mmi_json.txt'
    post = load(fpost)
    answer = load(fresp)
    result = load(fresult)
    store(post, answer, [], result, ftgt)
