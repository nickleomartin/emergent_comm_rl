random_config_dict = {
				## Data Generation
				"v_dim": 50,
				"training_n_dim": 1000,
				"testing_n_dim": 1000,
				"n_distractors": 4,

				## Language 
				"max_message_length": 2,
				"alphabet_size": 62,
				"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],

				## Speaker
				"speaker_lr": 0.01, #0.0001
				"speaker_dim": 50,

				## Listener
				"listener_lr": 0.01, #0.001
				"listener_dim": 50,

				## Training 
				"training_epoch": 10000,
				"batch_size": 32,

				## Train-split 
				"train_split_percent": 0.8,
}

visa_config_dict = {
				## Data Generation
				"n_distractors": 4,

				## Language 
				"max_message_length": 2,
				"alphabet_size": 62,
				"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],

				## Speaker
				"speaker_lr": 0.01, #0.0001
				"speaker_dim": 594,

				## Listener
				"listener_lr": 0.01, #0.001
				"listener_dim": 50,

				## Training 
				"training_epoch": 10000,
				"batch_size": 32,
				"n_batches": 100,

				## Train-split 
				"train_split_percent": 0.8,
}