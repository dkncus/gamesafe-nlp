import csv

datasets = ['./datasets/messages_scary.csv',
            './datasets/FORMATTEDdiscordchat1.csv',
            './datasets/FORMATTEDdiscordchat4.csv',
            './datasets/FORMATTEDLOLChatLog.csv',
            './datasets/FORMATTEDmoredota2_chat.csv',
            './datasets/FORMATTEDdota2_chat_messages.csv']

def is_ascii(s):
	return all(ord(c) < 128 for c in s)

# Create a list of messages from a CSV file.
def load_csv_to_list(csv_location):
	print('Reading File', csv_location)

	# List of messages to return
	messages = []

	# Load dataset
	with open(csv_location, newline='', encoding='utf-8', errors='ignore') as sample:

		# Create a CSV reader object delimited at commas
		reader = csv.reader(sample, delimiter=',')

		# For each row in the CSV that is being read from
		if csv_location == './datasets/messages_scary.csv':  # Messages_scary has 2 columns, the second of which is the message
			for i, row in enumerate(reader):
				# print(i, row[2])

				messages.append(row[1])  # Read into message list
		else:
			for i, row in enumerate(reader):
				# print(i, row[2])
				messages.append(row[0])  # Read into message list

	return messages

def normalise_text (text):
	text = text.lower() # lowercase
	text = text.replace(r"\#","") # replaces hashtags
	text = text.replace(r"http\S+","URL")  # remove URL addresses
	text = text.replace(r"@","")
	text = text.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
	text = text.replace("\s{2,}", " ")
	return text

with open('../../datasets/data_full.csv', 'w', newline='', encoding='utf-8', errors='ignore') as file:
	writer = csv.writer(file)
	writer.writerow(["Message", "Type"])
	for dataset in datasets:
		data = load_csv_to_list(dataset)
		if dataset != './datasets/messages_scary.csv':
			for message in data:
				if is_ascii(message):
					writer.writerow([normalise_text(message), 0])
		else:
			# Open the dataset object to read
			with open(dataset, newline='', encoding='utf-8', errors='ignore') as sample:
				# Create a CSV reader object delimited at commas
				reader = csv.reader(sample, delimiter=',')

				# For each row in the CSV that is being read from
				for row in reader:
					# print(i, row[2])
					if row[2] == row[3]:
						if is_ascii(row[1]):
							writer.writerow([normalise_text(row[1]), 1])
					else:
						if is_ascii(row[1]):
							writer.writerow([normalise_text(row[1]), 0])
							
				