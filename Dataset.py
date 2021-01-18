import os
import pickle
import numpy as np
# from time import time
# from datetime import datetime
import time, datetime
from tqdm import tqdm

class DataSet:
	def __init__(self, data_path='/home/zhengyi_ma/pcb/Data/',
				dialoglogfile='PChatbot_byuser_small', word2vec_path='/home/zhengyi_ma/pcb/Data/PChatbot.word2vec.200d.txt',word2vec_dim=200,
				limitation=10000000,max_history_len=15, batch_size=64, num_epoch=1 ,max_dec_steps=50):
	
		
		self.data_path = data_path
		self.in_path = os.path.join(data_path, dialoglogfile)
		print("start loading log file list...")
		self.filenames  =  sorted(os.listdir(self.in_path))
		print("loading log file list complete")
		# print(self.filenames[10])
		# print(self.filenames[100])

		self.word2vec_dim = word2vec_dim
		self.word2vec_path = word2vec_path
		self.limitation = limitation
		self.max_history_len = max_history_len
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.max_dec_steps = max_dec_steps
		
		
		# load word embedding
		self.word2emb_dict = {} # word -> nparray
		self.word2id = {"[PAD]":0,"[UNK]":1,"[START]":2,"[END]":3}
		print("start loading word embeddings...")
		with open(self.word2vec_path,"r") as f_wordemb:
			cnt = 0
			for line in f_wordemb:
				cnt += 1
				if cnt == 1:
					self.vocab_size = int(line.split()[0])
					# add [PAD] [UNK] [START] [END]
					self.W_init = np.random.uniform(-0.25, 0.25, ((self.vocab_size + 4), self.word2vec_dim)) # 直接用来初始化embedding矩阵
					continue
				else:
					line_list = line.split()
					word = line_list[0]
					vector = np.fromstring(" ".join(line_list[1:]), dtype=float, sep=' ')
					self.word2emb_dict[word] = vector
					# cnt = 2对应第一个词 W_init中下标=4
					self.W_init[cnt + 2] = vector
					self.word2id[word] = cnt + 2
		print("loading word embeddings complete")
		
	
	
	def init_dataset(self):

		self.p_train = [] # current post
		self.p_len_train = [] # current post length
		self.r_train = []  # current response 
		self.r_len_train = []  # current response length

	
	def trans_sentence_to_idx(self, sent): 
		# 将句子转成idx_list
		idx_p = []
		for w in sent.split(" "):
			if w in self.word2id:
				idx_p.append(self.word2id[w])
			else:
				idx_p.append(self.word2id['[UNK]'])
		return idx_p
	
	def can_as_data(self, r_time, last_r_time):
		if int(r_time) - int(last_r_time) > 1 * 60 * 10:
			return True
		else:
			return False

	def prepare_single_data(self, p, r, label="train"):
		# print([p],[r])
		

		idx_p = self.trans_sentence_to_idx(p)
		idx_r = self.trans_sentence_to_idx(r)

		if label == "train":
			self.p_train.append(idx_p)
			self.p_len_train.append(len(idx_p))
			self.r_train.append(idx_r)
			self.r_len_train.append(len(idx_r))
		

		# print([idx_p],[idx_r])



	def prepare_dataset(self): 
		# 往init_dataset里准备的容器里灌数据
		if not hasattr(self, "p_train"):
			self.init_dataset()
		
		user_id = 0
		print("There are %d users in the log directory" % (len(self.filenames)))
		for filename in tqdm(self.filenames):
			user_id += 1


			last_r_time = 0 # 上一条response的时间
			fhand = open(os.path.join(self.in_path,filename))
			
			for line in fhand:

				p, p_uid, p_time, r, r_uid, r_time, _, phase = line.strip().split("\t")
				if self.can_as_data(r_time, last_r_time): # 这条数据可以作为一条数据
					self.prepare_single_data(p, r) # 生成一条数据
				last_r_time = r_time
	
	def gen_batchs(self, perm):
		pair_train = len(self.p_train)
		for j in range(int(pair_train/self.batch_size)):
			train_start = j*self.batch_size
			train_end = (j+1)*self.batch_size
			posts = [self.p_train[item] for item in perm[train_start:train_end]]
			responds = [self.r_train[item] for item in perm[train_start:train_end]]

			# construct posts for batch
			posts_max_len = max([len(p) for p in posts])
			posts_batch = np.zeros((self.batch_size, posts_max_len), dtype=np.int32)
			posts_lens = np.zeros((self.batch_size), dtype=np.int32)
			posts_padding_mask = np.zeros((self.batch_size, posts_max_len), dtype=np.float32)

			for i, post in enumerate(posts):
				posts_lens[i] = len(post)
				for j in range(len(post)):
					posts_batch[i][j] = post[j]
					posts_padding_mask[i][j] = 1

			# construct responses for batch
			
			dec_batch = np.zeros((self.batch_size, self.max_dec_steps), dtype=np.int32)
			target_batch = np.zeros((self.batch_size, self.max_dec_steps), dtype=np.int32)
			dec_padding_mask = np.zeros((self.batch_size, self.max_dec_steps), dtype=np.float32)
			for i, response in enumerate(responds):

				inp = [self.word2id["[START]"]] + response
				target = response
				if len(inp) > self.max_dec_steps:  # truncate 超长度了 截断
					inp = inp[:self.max_dec_steps]
					target = target[:self.max_dec_steps]  # no end_token
				else:  # no truncation
					target.append(self.word2id["[END]"])  # end token 没有超长度 就得decode至[END]
				assert len(inp) == len(target)
				# for j in range(len(post)):
				for j in range(len(inp)):
					dec_batch[i][j] = inp[j]
					target_batch[i][j] = target[j]
					dec_padding_mask[i][j] = 1
			

			
			yield [posts_batch, posts_lens, posts_padding_mask, dec_batch, target_batch, dec_padding_mask]




	def gen_epochs(self):
		pair_train = len(self.p_train)
		for i in range(self.num_epoch):
			perm = np.random.permutation(pair_train)
			yield self.gen_batchs(perm)

					



d = DataSet()
d.init_dataset()
d.prepare_dataset()
print("dataset.p_train nums: %d" % (len(d.p_train)))

# print(len(d.p_train))
# print(d.word2emb_dict['我们'])

for idx, epoch in enumerate(d.gen_epochs()):
	batch_nums = int(len(d.p_train)/d.batch_size)
	print("epoch %d start, there are %d batches" % (idx, batch_nums ))
	batch_cnt = 0
	for p_train, p_len_train, p_padding_mask, dec_train, target_train, dec_padding_mask in epoch:
		print("epoch %d: gen batch %d/%d " % (idx, batch_cnt, batch_nums))
		batch_cnt += 1
		# print("1")
		# print(posts_train[5])
		# print(posts_len_train[5])
		# print(posts_padding_mask[5])
		# exit(0)
		# print(dec_train)
		# print(target_train)
		# print(dec_padding_mask)		

				
				





	