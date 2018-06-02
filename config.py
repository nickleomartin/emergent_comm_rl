config_dict = {
				## Data Generation
				"v_dim": 50,
				"n_dim": 1000,
				"n_distractors": 4,

				## Language 
				"max_message_length": 2,
				"alphabet_size": 100,
				"alphabet": [str(c) for c in range(100)],

				## Speaker
				"speaker_lr": 0.0001,
				"speaker_dim": 50,

				## Listener
				"listener_lr": 0.001,
				"listener_dim": 50,

				## Training 
				"training_epoch": 10000,
				"batch_size": 32,
}