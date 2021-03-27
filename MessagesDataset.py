# Machine Learning Libraries
import torch
from torch.utils.data import Dataset
import pandas as pd

# NLP Libraries
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')  # Define list of stopwords

# Class for loading the messages dataset
class MessagesDataset(Dataset):
	def __init__(self, csv_file, SENTENCE_LENGTH):
		print('Loading Data from file', csv_file)
		self.messages_frame = pd.read_csv(csv_file)
		self.embedding_model = gensim.models.KeyedVectors.load('./models/model.wordvectors')
		self.SENTENCE_LENGTH = SENTENCE_LENGTH
		
	def __len__(self):
		return len(self.messages_frame.index)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		# Split the message into words, get appropriate label
		message = []
		message_words = word_tokenize(self.messages_frame.iloc[idx, 0])
		label = self.messages_frame.iloc[idx, 1]
		
		# Create a list of message words that include only words in the vocab and not in the stopwords
		for word in message_words:
			if word not in stop_words and word in self.embedding_model.wv.vocab:
				message.append(self.embedding_model.wv.vocab[word].index)
		
		# If the message is less than a given number of words long
		if len(message) < self.SENTENCE_LENGTH:
			
			# Add a zero for each missing word to pad the vector
			pad_zeroes = self.SENTENCE_LENGTH - len(message)
			for i in range(pad_zeroes):
				message.append(0)
		
		# If the message is greater than that number of words
		elif len(message) > self.SENTENCE_LENGTH:
			# Trim the message
			message = message[:self.SENTENCE_LENGTH]
		
		# Create sample dictionary containing appropriate data and return it
		sample = {
			'message': message,
			'label': [int(label)],
		}
		
		return sample
