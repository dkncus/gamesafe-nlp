import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import gensim
import torch

class LanguageClassifier(nn.Module):
	# Define Network Structure
	def __init__(self, BATCH_SIZE, SENTENCE_LENGTH, EMBEDDING_MODEL):
		super().__init__()
		# Define network-specific data
		self.BATCH_SIZE = BATCH_SIZE
		self.SENTENCE_LENGTH = SENTENCE_LENGTH
		
		# Load pretrained text embeddings model
		self.embedding_model = gensim.models.KeyedVectors.load(EMBEDDING_MODEL)
		weights = torch.FloatTensor(self.embedding_model.wv.vectors)
		
		# Create word embedding layer
		self.embedding = nn.Embedding.from_pretrained(weights)
		
		# Create LSTM Layer
		self.lstm = nn.LSTM(input_size=SENTENCE_LENGTH, hidden_size=32, num_layers=1, batch_first=True)  # lstm
		
		# Create Labeling layer
		self.fc = nn.Sequential(
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Sigmoid(),
		) 

	# Define forward propagation step
	def forward(self, x):
		# Convert the input word to a tensor through text embedding model
		x = self.embedding(x)
		
		# Create initial vectors for cell states
		h_0 = Variable(torch.zeros(1, x.size(0), 32))
		c_0 = Variable(torch.zeros(1, x.size(0), 32))
		
		# Get output from LSTM Module
		output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
		
		# Reshape data for the final layers
		hn = hn.view(-1, 32)
		
		# Send data through fully connected sequential layer
		out = self.fc(hn)
		
		# Reshape to backprop
		return out