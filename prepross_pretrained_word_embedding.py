import   codecs
import numpy as np
from tqdm import tqdm
def load_dense_drop_repeat(path, vocab_fp, embedding_fp):
    vocab_size, size = 0, 0
    count = 0
    with codecs.open(path, "r", "utf-8") as f, open(vocab_fp, "w") as fp:
        first_line = True
        for line in tqdm(f):
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.rstrip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue

            vec = line.strip().split()
            matrix[count, :] = np.array([float(x) for x in vec[1:]])
            count += 1
            fp.write(vec[0] + '\n')
        np.save(embedding_fp, matrix)
        return count

if __name__ == "__main__":
    path = '../data/pretrained/legal/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    vocab_fp = '../data/pretrained/legal/vocab_file'
    embedding_fp = '../data/pretrained/legal/embedding_file.npy'
    cnt = load_dense_drop_repeat(path, vocab_fp, embedding_fp)
    # emb = np.load(embedding_fp)
    # print('emb的shape:', emb.shape)
    print(cnt)