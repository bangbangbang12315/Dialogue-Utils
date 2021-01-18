import sentencepiece as spm
import os
dir_path = '../data/Pdata/reddit'
vocab_file = 'vocab.txt'

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(dir_path, 'spm.model'))

with open(os.path.join(dir_path, vocab_file), 'w') as fv:
    for id in range(sp.get_piece_size()):
        fv.write(sp.id_to_piece(id) + ' ' + str(id) + '\n')