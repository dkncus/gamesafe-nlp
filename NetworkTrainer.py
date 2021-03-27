# Natural Language Processing Libraries
from nltk.corpus import stopwords
stop_words = stopwords.words('english')  # Define list of stopwords

# Machine Learning Libraries
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert

# Data Visualization Libraries
import matplotlib.pyplot as plt

# Self-defined Libraries (see MessagesDataset.py, LanguageClassifier.py, and ProgressBar.py)
from MessagesDataset import MessagesDataset
from NeuralNetworks import LanguageClassifier
from ProgressBar import ProgressBar
from statistics import mean

class NetworkTrainer():
	# Initialization Method
	def __init__(self, train_csv, test_csv, dev_csv, save_name, EMBEDDING_MODEL='./models/model.wordvectors', EPOCHS=1, LEARNING_RATE=10e-5, BATCH_SIZE=64, SENTENCE_LENGTH=10, BATCH_DIAGNOSTICS=1):
		# Load Dataset
		self.dataset = MessagesDataset(train_csv, SENTENCE_LENGTH)
		self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=default_convert)
		self.DATASET_SIZE = len(self.dataset)
		self.train_csv = train_csv
		self.test_csv = test_csv
		self.dev_csv = dev_csv
		
		# Hyperparameters
		self.EPOCHS = EPOCHS
		self.LEARNING_RATE = LEARNING_RATE
		self.BATCH_SIZE = BATCH_SIZE
		self.SENTENCE_LENGTH = SENTENCE_LENGTH
		self.BATCH_DIAGNOSTICS = BATCH_DIAGNOSTICS
		self.SAVE_NAME = save_name
		
		# Network Parameters
		self.network = LanguageClassifier(BATCH_SIZE, SENTENCE_LENGTH, EMBEDDING_MODEL)
		self.criterion = torch.nn.BCELoss()
		self.optimizer = optim.Adam(self.network.parameters(), lr=self.LEARNING_RATE)

	# Train the neural network
	def train(self):
		loss_statistics = []
		acc_statistics = []

		# TRAINING EPOCHS
		for epoch in range(self.EPOCHS):
			# Diagnostic Data
			print('EPOCH', epoch + 1, '/', self.EPOCHS)

			# Progress Bar object
			pb = ProgressBar()

			# Statistics
			loss_data = []
			acc_data = []

			# For each training example 
			for i, batch in enumerate(self.dataloader):
				# Create message and label tensors
				msg_tensor, lbl_tensor = self.prep_tensors(batch)

				# Send data through neural net, calculate loss, and back propagate
				self.optimizer.zero_grad()
				output = self.network(msg_tensor)
				loss = self.criterion(output, lbl_tensor.type_as(output))
				loss.backward()
				self.optimizer.step()

				# Calculate Statistics
				loss_data.append(float(loss))
				acc_data.append(self.calc_acc(output, lbl_tensor))

				# Print batch diagnostics
				if i % self.BATCH_DIAGNOSTICS == 0:
					prefix = f'Epoch Progress'
					suffix = f'[Batch {i}/{self.DATASET_SIZE // self.BATCH_SIZE}] ({self.BATCH_SIZE}) ' \
					         f'| Loss : {round(mean(loss_data), 5)} ' \
					         f'| Acc : {round(mean(acc_data) * 100, 2)}%'
					pb.printProgressBar(i + 1, self.DATASET_SIZE // self.BATCH_SIZE, prefix=prefix, suffix=suffix, length=30)

			loss_statistics.append(loss_data)
			acc_statistics.append(acc_data)
			print()

		self.display_statistics(loss_statistics, acc_statistics)

	# Save the model to a filename
	def save_model(self):
		torch.save(self.network.state_dict(), f'./models/{self.SAVE_NAME}.torch')

	# Run test dataset on network
	def dev(self):
		# Load train, test, and dev data
		print('Loading Training Data...')
		data = MessagesDataset(self.dev_csv, self.SENTENCE_LENGTH)
		dataloader = DataLoader(data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0)

		acc_data = []
		for batch in dataloader:
			msg_tensor, lbl_tensor = self.prep_tensors(batch, test=True)
			output = self.network(msg_tensor)
			acc_data.append(self.calc_acc(output, lbl_tensor))

		print(f'Dev set accuracy : {round(mean(acc_data), 3)}%')
	
	# Show accuracy and loss curves during training
	def display_statistics(self, loss_statistics, acc_statistics):
		# Create X axis
		print(len(loss_statistics[0]))

		# Calculate loss Statistics
		loss_statistics_avg = []
		for l in loss_statistics:
			loss_statistics_avg.append(self.calc_moving_avg(l, 100))

		x_axis = [x for x in range(len(loss_statistics_avg[0]))]
		for i, stats in enumerate(loss_statistics_avg):
			plt.plot(x_axis, stats, label=f'Epoch {i + 1}')

		plt.xlabel('Training Example')
		plt.ylabel('Loss')
		plt.title('Loss per Epoch')
		plt.legend()

		# Display a figure.
		plt.show()

		# Clear the figure
		plt.clf()

		# Display Accuracy Statistics
		acc_statistics_avg = []
		for l in acc_statistics:
			acc_statistics_avg.append(self.calc_moving_avg(l, 100))

		x_axis = [x for x in range(len(acc_statistics_avg[0]))]
		for i, stats in enumerate(acc_statistics_avg):
			plt.plot(x_axis, stats, label=f'Epoch {i + 1}')

		plt.xlabel('Training Example')
		plt.ylabel('Accuracy')
		plt.title('Accuracy per Epoch')
		plt.legend()

		# Display a figure.
		plt.show()

	# Calculates the moving average of a list
	def calc_moving_avg(self, data, N):
		cumsum = [0]
		moving_aves = []

		for i, x in enumerate(data, 1):
			cumsum.append(cumsum[i - 1] + x)
			if i >= N:
				moving_ave = (cumsum[i] - cumsum[i - N]) / N
				# can do stuff with moving_ave here
				moving_aves.append(moving_ave)

		return moving_aves

	# Prepare tensors to be fed into Neural Network with Batching
	def prep_tensors(self, batch, test=False):
		if not test:
			msg_tensor = []
			lbl_tensor = []
			
			for data_object in batch:
				msg_tensor.append(data_object['message'])
				lbl_tensor.append(data_object['label'])
	
			msg_tensor = torch.tensor(msg_tensor)
			lbl_tensor = torch.tensor(lbl_tensor)
	
			return msg_tensor, lbl_tensor
		else:
			msg_tensor = [batch['message'][0]]
			lbl_tensor = [batch['label'][0]]
			
			print(msg_tensor)
			print(lbl_tensor)
			
			return msg_tensor, lbl_tensor
			

	# Calculate the accuracy of a given batch
	def calc_acc(self, output, label_tensor):
		output = output.tolist()
		label_tensor = label_tensor.tolist()
		
		count_correct = 0
		for i, label in enumerate(output):
			calc_val = 0
			if label[0] >= 0.5:
				calc_val = 1

			if calc_val == label_tensor[i][0]:
				count_correct += 1

		return count_correct / len(output)