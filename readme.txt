The test_on_data folder contains files that were used during testing phase. Thus, they can be used to regenerate the results provided in the slides.
Execution is same as the other folder.

File 1: anomaly_data.py # generates feature vectors from data
File 2: anomaly_train.py # Finds optimal nu and gamma. Also, generates roc_values.txt file.
File 3: severity_score_batch_generator.py
	Input to file 3: Best value corresponding to recall and auc score, usually the top value in roc_values.txt.
	If better more balanced high scores are available across precision, recall, accuracy and auc, choose the corresponding threshold. 
	# Generates batches of data: 1. Batches of size 100 in folder features_rbf_batch and 2. Batches of data per second in folder features_rbf_per_sec
	
File 4: Main.py, execute it to get interface which gives malicious trades and malicious traders. 
	Arguments: no of trades to observe per cluster (default = 5)
		 : no of analysts (default = 5)
		 : severity threshold (default = 5)
		 : mini batches to be left out (default = 0) # If the last mini batch has no. of tuples < no of analysts, it raises a warning.
							       To not take that mini batch into account, give '1' as argument
	Note*: no of trades to observe per cluster*no of analysts < batch_size of mini batch
		For eg. Since mini batch size is 100, no of analysts(<=10) and no of trades to observe per cluster(<=10)  
