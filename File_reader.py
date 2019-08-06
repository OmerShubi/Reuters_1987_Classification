import pickle
import re
import math

class File_reader:
	def __init__(self, data_articles):
		self.data_articles = data_articles
		self.number_of_docs = len(self.data)
		self.df = {}
		self.words = {}
		self.stop_words = []
		self.create_stop_words_list()
		self.create_words_bank()
		#self.inv_words = {v: k for k, v in self.words.items()}

	def create_stop_words_list(self):
		with open('./stop_words.txt') as stop_words_file:
			for stop_word in stop_words_file:
				self.stop_words.append(stop_word.rstrip())

	def pre_process(self, word):
		"""
		Preprocesses a word by turning it to lower case,
		removes some punctuation and returns the word or
		empty string if it is a stop word
		:param word: string
		:return: updated word - string
		"""
		strip_specail_chars = re.compile("[^A-Za-z ]+")
		word = word.lower().replace("<br />"," ")
		word = re.sub(strip_specail_chars, "", word.lower())
		# returns empty string if word is a stop word
		if word in self.stop_words:
			return ''
		return word

	def create_words_bank(self):
		"""
		create_words_bank from all the articles
		:return:
		"""
		index = 0
		for article in self.data_articles:
			seen_in_this_article = []
			for word in article['text'].split():
				word = self.pre_process_word(word)
				if word == '':
					continue
				if word not in self.df:
					self.df[word] = 1  # document frequency
					seen_in_this_article.append(word)
				if word not in seen_in_this_article:
					self.df[word] += 1
					seen_in_this_article.append(word)
				if word not in self.words.keys():  # if the word doesnt already exists in the words dictionary
					self.words[word] = index  # add it
					index += 1


	def build_set_tfidf(self, file_to_vector):
		"""
		Builds the data vector using tfidf format
		:param file_to_vector: the file to be processed
		:return: the file in vector form, using tfidf format
		"""
		doc_set = []
		for article in self.data_articles:
			vec = len(self.words) * [0, ]
			for word in article['text'].split():
				word = self.pre_process_word(word)
				if word == '':
					continue
				vec[self.words[word]] += 1
				# After iterating over all words we now have the tf and can store words in tfidf format
				for word in self.words.keys():
					if vec[self.words[word]] == 0:
						continue
					else:
						vec[self.words[word]] = vec[self.words[word]]\
												* math.log((self.number_of_docs / self.df[word]), 10)
			#TODO: labels pull
			doc_set.append(vec)
		return doc_set


	def unpickle(file):
		with open(file, 'rb') as f:
			data = pickle.load(f, encoding='bytes')
		return data








	def remove_stopwords

	def tf_idf




