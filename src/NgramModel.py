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
	
	# def ngram_prob(self, seq, order, k):
	# 	"""
	# 		Compute p(w_t|w_{t-order+1:t})
	# 	"""
	# 	if len(seq)==1:
	# 		pwt = self.ngram_counter[1].get(seq, 0) + k 
	# 		p = self.ngram_sum[1] + k*self.type_size
	# 		return pwt/p
		
	# 	elif 1<len(seq)<order:
	# 		order = len(seq)
	# 	wt_1 = seq[:order-1]
	# 	joint = (self.ngram_counter[order].get(seq, 0) + k)
	# 	pwt_1 = (self.ngram_counter[order-1].get(wt_1, 0)+k*self.type_size)
	# 	return joint/pwt_1
	def ngram_prob(self, seq, order,  lambdas, type='default',):
		"""
			Compute p(w_t|w_{t-order+1:t})
		"""
		if type=="default":
			if len(seq)!= order:
				return 1e-8

			wt_1 = seq[:order-1]
			joint = (self.ngram_counter[order].get(seq, 0))
			pwt_1 = (self.ngram_counter[order-1].get(wt_1, 0))
			if joint ==0 or pwt_1 ==0:
				return 1e-8
			return joint/pwt_1

		elif type == "interpolate":
			if sum(lambdas) != 1:
				raise ValueError("lambdas must sum to 1")

			log_prob = 0
			for i in range(1, order + 1):
				joint = np.array([self.ngram_counter[i].get(seq[:i], 0)])
				log_prob += lambdas[i - 1] * order_log_prob
			return log_prob

		elif type == "backoff":
			pass
	

	def sent_logprob(self, sentence, order, k=1e-8):
		logprob, nbr = 0, 0
		for seq in sentence.get_ngram(order):
			logprob +=np.log(self.ngram_prob(seq, order, k))
			nbr+=1
		return logprob, nbr


	def interpolate(self, sentence, order, lambdas):
		"""
			Compute the interpolated probability of a sentence 
			using n-gram probabilities from different orders.
		"""
		if sum(lambdas) != 1:
			raise ValueError("lambdas must sum to 1")

		log_prob = 0
		for i in range(1, order + 1):
			order_log_prob, _ = self.sent_logprob(sentence, i, k=1e-8)
			log_prob += lambdas[i - 1] * order_log_prob
		return log_prob

	def interpolate_bo(self, sentence, order):
		pass	

	def generate(self, start, max_len=100, k=1e-8):
		text = '<s> '+ start
		context = ['<s>']
		context.extend(start)
		context = context[-self.orders:]
		for _ in range(max_len):
			probs = self.__get_next_probs(context, k)
			next_word = self.__sample_word(probs)
			context = context[1:] + [next_word]
			text += ''.join(next_word)
		return text
	
	def __get_next_probs(self, context, k):
		probs = np.zeros(self.type_size)
		for i in range(len(self.type)):
			seq= tuple( context + [self.type[i]])
			probs[i] = self.ngram_prob(seq, self.orders, k)
		return probs/sum(probs)

	def __sample_word(self, probs):
		idx = np.random.multinomial(1, probs).argmax() 
		return self.type[idx]
	
	def __repr__(self):
		return self.name
	