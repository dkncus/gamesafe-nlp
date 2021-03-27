import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
View tensor data

			except Exception as e:
				print(data['message_words'])
				msg = msg_tensor.tolist()
				for obj in msg:
					x = dataset.embedding_model.wv.index2word[obj[0]]
					print(obj, x)
				print(lbl_tensor)
				print(e)
				exit(0)
'''

'''
LSTM Information
https://cnvrg.io/pytorch-lstm/?gclid=Cj0KCQjwjPaCBhDkARIsAISZN7S7uggC0XHu3gKn5jxi5YTgYF8Pu4JRf6yJS-nKXDdwGlfMjqCoTXoaAhUGEALw_wcB
'''

'''
# Concatenate 2 frames and save them as a CSV

df_1 = pd.read_csv('./messages_scary_with_window.csv')
df_2 = pd.read_csv('./messages_normal_with_window.csv')

frames = [df_1, df_2]

result = pd.concat(frames)

result.to_csv('./training/data_windowed.csv', index = False)
'''

"""
# Create a new CSV from the scary messages of messages 
stop_words = stopwords.words('english')  # Define list of stopwords

WINDOW_SIZE = 10

df = pd.read_csv('./longestfirst.csv', encoding='latin-1')
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

messages = []

# For each row in the CSV
for row in df.iterrows():
	# strip punctuation from sentence
	no_punct = ""
	for char in row[1]['Message']:
		if char not in punctuations:
			no_punct = no_punct + char
	
	# Tokenize the cleaned sentence
	words = word_tokenize(no_punct)
	
	# Add a row for every sliding window
	if len(words) >= WINDOW_SIZE:
		for i in range(len(words)- WINDOW_SIZE):
			window = words[i:i+WINDOW_SIZE]
			string = ''
			for i, str in enumerate(window):
				if i < WINDOW_SIZE - 1:
					string += str + ' '
				else: 
					string += str
			
			messages.append({'Message': string, 'Type': 1})	
			print(string)
	elif len(words) >= 4:
		string = ''
		for i, str in enumerate(words):
			if i < WINDOW_SIZE - 1:
				string += str + ' '
			else:
				string += str
		messages.append({'Message': string, 'Type': 1})	
		print(string)

df_new = pd.DataFrame(messages, columns=['Message', 'Type'])
df_new.to_csv('./messages_normal_with_window.csv', index = False)
print(len(messages))
"""

'''
# Get the dataframe sorted by longest message

df = pd.read_csv('./datasets/FORMATTEDLOLChatLog.csv', encoding='latin-1')
x = df.reindex(df['message'].str.len().sort_values().index)
x = x.reindex(index = x.index[::-1])
x.to_csv('./longestfirst_LoL.csv', index = False)
'''

'''
for splitting datasets amongst various .csv files

import pandas as pd

df = pd.read_csv('./training/data_full_clean.csv')

train = df.sample(frac=0.98,random_state = 200) #random state is a seed value
devtest = df.drop(train.index)
test = devtest.sample(frac=0.5, random_state=200)
dev = devtest.drop(test.index)

train.to_csv('./training/data_train.csv', index = False)
test.to_csv('./training/data_test.csv', index = False)
dev.to_csv('./training/data_dev.csv', index = False)
'''
