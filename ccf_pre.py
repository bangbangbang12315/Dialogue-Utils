import json
from functools import reduce
f = '../data/input_B/DuConv/test_2.txt'
fp_out = '../data/input_B_1.txt'
with open(f, 'r') as fp, open(fp_out, 'w') as fq:
    for line in fp:
        data = json.loads(line)
        if len(data['history']) == 0:
            x = reduce(lambda x1, x2: x1 + x2, data['goal'])
        else:
            x = data['history']
        print(x)
        fq.write(' '.join(x) + '\n')