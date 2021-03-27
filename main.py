from NetworkTrainer import NetworkTrainer

if __name__ == '__main__':
	# Create a neural network training object
	trainer = NetworkTrainer(
		'training/data_dev.csv',      # Training Dataset Location
		'training/data_dev.csv',        # Development Dataset Location
		'training/data_test.csv',       # Testing Dataset Location
		'model_3_27',                   # Filename to save the model as once finished training - ./models/{SAVE_NAME}.torch
		'./models/model.wordvectors',   # Location for loading the word vector model
		EPOCHS = 3,                     # Number of Epochs to train the model on
		LEARNING_RATE = 10e-5,          # α - Learning rate of the model
		BATCH_SIZE = 128,               # Size of batch for vectorization (Most efficient in powers of 2)
		SENTENCE_LENGTH = 10,           # Length of sentence (Length of input tensor)
		BATCH_DIAGNOSTICS = 1           # Display Batch Diagnostics
	)
	
	# Train the Neural Network
	print('Training Neural Network...')
	trainer.train()
	print('Training Complete.')
	
	# Show Statistics on Dev Set
	# trainer.dev()
	
	# Save to model
	trainer.save_model()

'''
 ///,        ////
 \  /,      /  >.
  \  /,   _/  /.
   \_  /_/   /.
    \__/_   <
    /<<< \_\_
   /,)^>>_._ \
   (/   \\ /\\\
        // ````
       ((`
       
       David Kubala
'''