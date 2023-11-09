# Model-Free-Q-Learning
Model Free Q-Learning

This Python script implements a model-free reinforcement learning strategy using Q-learning to derive a policy from a given dataset in CSV format. The script begins by importing essential libraries and modules and then proceeds to prepare the data by reading the CSV file into a pandas DataFrame. The Q-learning algorithm, defined in the q_learning function, is employed to iteratively update a Q-matrix, which represents Q-values for state-action pairs, creating a matrix with actions as the rows and states as the column value. This iterative process refines the policy by learning the best actions to take in different states. The compute function orchestrates the data processing and policy computation, generating a policy file based on the learned policy. The main function serves as the entry point, verifying the correct usage of command-line arguments and measuring the script's total runtime. Ultimately, this script is designed to be executed from the command line, using a provided CSV file to learn a policy through Q-learning and save it in an output file for future use.

Small.csv

Training runtime: 10.457550048828125 seconds

Total runtime: 10.469928979873657 seconds

Medium.csv

Training runtime: 20.629810333251953 seconds

Total runtime: 20.66704511642456 seconds

Large.csv

Training runtime: 20.993757247924805 seconds

Total runtime: 21.12453007698059 seconds

The runtime differences in the provided code for different CSV files, namely small.csv, medium.csv, and large.csv, can be attributed to the varying size and complexity of the datasets. The small dataset, with 100 states and 4 actions, has the shortest training runtime of approximately 10.46 seconds, as the Q-learning algorithm converges quickly on smaller datasets. In contrast, the medium dataset, with 50,000 states and 7 actions, takes approximately 20.63 seconds to train, as it involves a larger state-action space and thus requires more iterations for convergence. The large dataset, with 312,020 states and 9 actions, has a training runtime of approximately 20.99 seconds, similar to the medium dataset, despite its larger state space, due to factors like algorithm convergence properties and problem complexity. These runtime variations highlight the impact of dataset size and complexity on the training time of the Q-learning algorithm.
