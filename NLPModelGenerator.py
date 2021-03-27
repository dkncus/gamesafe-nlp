# NLTK Natural Language Processing Models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim.models import Word2Vec
import csv

class NLPModelGenerator():
	# Initialization Method
	def __init__(self, train=False, load=True):
		# Train the model
		if train:
			print('Loading Messages...')
			messages = self.load_csv_to_list('./data_full.csv')
			print('Creating embedding model...')
			self.model = self.create_model(messages)
			self.model = self.pickle_model(self.model, '../nlang/models/model.wordvectors')
			print('Model compiled and saved.')
		# Load a preexisting model
		if load:
			print('Loading preexisting model...')
			self.model = gensim.models.KeyedVectors.load('./models/model.wordvectors')
			print('Model loaded.')
		
	# Create a list of messages from a CSV file.
	def load_csv_to_list(self, csv_location):
		print('Reading File', csv_location)
	
		# List of messages to return
		messages = []
	
		# Load dataset
		with open(csv_location, newline='', encoding='utf-8', errors='ignore') as sample:
	
			# Create a CSV reader object delimited at commas
			reader = csv.reader(sample, delimiter=',')
			
			for i, row in enumerate(reader):
				# print(i, row[2])
				messages.append(row[0])  # Read into message list
	
		return messages
	
	# Create and return a trained model.
	def create_model(self, messages):
		# Create data object (Lists of words)
		data = []
	
		stop = stopwords.words('english')
	
		# For each message
		num_messages = len(messages)
		for i, message in enumerate(messages):
			# Diagnostic Information
			print('<', i, '/', num_messages, ">:", message)
	
			# Tokenize the sentence into a list of words
			temp = [token for token in word_tokenize(message) if not token in stop]
	
			# Append a list of words as a message to the dataset
			data.append(temp)
	
		# Create the word embedding model with Word2Vec algorithm
		print("Generating word embedding model...")
		model = gensim.models.Word2Vec(data, min_count=20, size=10, window=5, sg=1, workers=3)
		print("Embedding model created.")
	
		return model
	
	# Save the model to a file
	def pickle_model(self, model, filename):
		model.save(fname_or_handle=filename)
		return model

if __name__ == '__main__':
	# List of datasets
	nlp = NLPModelGenerator(train=True, load=False)
	
	# For each of these words
	words = ['lol', 'crazy', 'talk', 'game']

	# Print diagnostic data about words similar to this word
	for word in words:
		print('Analysis for', word)
		print(nlp.model[word].tolist())
		similar = nlp.model.most_similar(word)[:10]
		for i, item in enumerate(similar):
			print('\t', i, ':', item)