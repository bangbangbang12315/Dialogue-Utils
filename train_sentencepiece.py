import sentencepiece as spm
spm.SentencePieceTrainer.train('--input=../data/Pdata/reddit/sentences.txt --model_prefix=../data/Pdata/reddit/spm_8000 --vocab_size=8000 --model_type=bpe')