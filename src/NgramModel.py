from src.Tokenizer import Tokenizer
import numpy as np
from src.utils import sort_dict


class NGModel:
	def __init__(self, file_name: str, type:list, name: str, orders: int = 1):
		self.orders = self.__set_order(orders)
		self.name = name
		self.type = type
		self.type_size = len(self.type)
		self.token = Tokenizer(file_name)
		self.ngram_counter =  0
		self.ngram_sum = 0

		self.__set_count_sum()
	
	def __set_count_sum(self):
		self.ngram_counter, self.ngram_sum = self.__get_count()

	def __set_order(self, order):
		"""
			Handle order to be an int and >1
		"""
		if not isinstance(order, int):
			raise ValueError("order must be an insatnce of int")
		if order<1:
			raise ValueError("order must be greater than 0")
		return order

	def __counter(self, tokens, order):
		counters = {}
		L = len(tokens) - order
		for i in range(L):
			current = tuple(tokens[i * order:(i + 1) * order])
			if counters.get(current):
				counters[current] += 1
			else:
				counters[current] = 1
		return counters

	def __get_count(self):
		"""
			Compute all C(w_{1:i}), for i<= order
		"""
		counters = {}
		sums = {}
		token_list = list(self.token)
		for order in range(1, self.orders + 1):
			_tmp = self.__counter(token_list, order)
			counters[order] = sort_dict(_tmp)
			sums[order] = sum(_tmp.values())
		return counters, sums
	
	def ngram_prob(self, seq, order, k):
		"""
			Compute p(w_t|w_{t-order+1:t})
		"""
		if len(seq)==1:
			pwt = self.ngram_counter[1].get(seq, 0) + k 
			p = self.ngram_sum[1] + k*self.type_size
			return pwt/p
		
		elif 1<len(seq)<order:
			order = len(seq)
		wt_1 = seq[:order-1]
		joint = (self.ngram_counter[order].get(seq, 0) + k)
		pwt_1 = (self.ngram_counter[order-1].get(wt_1, 0)+k*self.type_size)
		return joint/pwt_1
	

	def sent_logprob(self, sentence, order, k=1e-8):
		logprob, nbr = 0, 0
		for seq in sentence.get_ngram(order):
			logprob +=np.log(self.ngram_prob(seq, order, k))
			nbr+=1
		return logprob, nbr

		




	# def generate(self, start, max_len=100, smoothing=0):
	# 	text = '<s>'+start
	# 	tokens = ["<s>", start]
	# 	for _ in range(max_len):
	# 		tokens = tokens[-(self.orders -2):]
	# 		probs = self.__get_next_probs(tokens, smoothing)
	# 		next_word = self.__sample_word(probs)
	# 		tokens.append(next_word)
	# 		text += ''.join(next_word)
	# 	return text
	
	# def __get_next_probs(self, tokens, smoothing):
	# 	order = self.orders
	# 	context= tuple(tokens[-(self.orders -2):])
	# 	pcontext = self.log_joints[self.orders -1].get(context, 0) + smoothing * len(self.vocab) 
	# 	probs = np.zeros((len(self.vocab),))
		
	# 	for i in range(len(self.vocab)):
	# 		joint = tuple(tokens[-(self.orders -2):] + [self.vocab[i]])
	# 		pjoint= self.log_joints[self.orders].get(joint, 0) + smoothing
	# 		probs[i] = pjoint/pcontext
	# 	# print(sum(probs))
	# 	return probs

	# def __sample_word(self, probs):
	# 	idx = np.random.multinomial(1, probs).argmax() 
	# 	return self.vocab[idx]
	
	# def perplexity(self, text, order, smoothing=0):
	# 	token = ['<s>']
	# 	token.extend(text)
	# 	token +=['</s>']
	# 	N = len(token)
	# 	P = 1
	# 	norm = smoothing*len(self.vocab)
	# 	for i in range(N-1):
	# 		joint = token[i*(order):(i+1)*order]
	# 		den = joint[:-1]
	# 		p = (self.log_joints[order].get(tuple(joint), 0) + smoothing)
	# 		p /= (self.log_joints[order -1].get(tuple(den), 0) + norm)
	# 		P *=(1/p)
	# 	return P**(1/N)
		
	def __repr__(self):
		return self.name
	