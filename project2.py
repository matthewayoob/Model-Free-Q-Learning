# Import necessary libraries and modules
import sys
import numpy as np
import pandas as pd
import time

# Function for initial data preprocessing
def initial_work(csv_name):
    # Read data from the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_name)
    # Get the column names
    names = df.columns
    # Convert the DataFrame to a NumPy array
    dfnp = df.to_numpy()
    print("Column names:", names)
    return dfnp

# Q-learning algorithm
def q_learning(dfnp):
    # Hyperparameters for Q-learning
    # Small dataset configuration
    # gamma = 0.95  # Discount factor
    # alpha = 0.01  # Learning rate
    # states_amt = 100  # Number of states
    # actions_amt = 4  # Number of possible actions

    #medium
    # gamma = 1
    # alpha = 0.01
    # states_amt = 50000
    # actions_amt = 7
    #large
    gamma = 0.95
    alpha = 0.01
    states_amt = 312020
    actions_amt = 9
    
    # Initialize the Q-matrix with zeros
    Q_matrix = np.zeros((states_amt, actions_amt))
    print("Q-matrix:", Q_matrix)
    
    # Q-learning iterations
    for i in range(100):
        for row in dfnp:
            s, a, r, sp = row  # State, action, reward, next state
            # Q-value update using the Q-learning equation
            Q_matrix[s - 1][a - 1] += alpha * (r + gamma * np.max(Q_matrix[sp - 1]) - Q_matrix[s - 1][a - 1])

    return Q_matrix

# Function to process and save results
def compute(csv_name):
    output = str(csv_name)[:-4] + ".policy"  # Output policy file name
    print("Input CSV file:", csv_name)
    
    # Perform initial data preprocessing
    dfnp = initial_work(csv_name)
    
    q_start_time = time.time()
    res = q_learning(dfnp)
    q_elapsed_time = time.time() - q_start_time
    
    # Save the learned policy to a file
    file_population(res, output)
    
    return dfnp, res, q_elapsed_time

# Function to save the policy as a text file
def file_population(Q_matrix, filepath):
    # Extract the best actions (policy) from the Q-matrix
    np_res = np.argmax(Q_matrix, axis=1) + 1
    # Save the policy to the specified file
    np.savetxt(filepath, np_res, fmt='%d')

# Main function to execute the code
def main():
    if len(sys.argv) != 2:
        raise Exception("Usage: python project1.py <infile>.csv -> <infile>.gph")
    
    # Get the input CSV filename from command line arguments
    inputfilename = sys.argv[1]
    
    start_time = time.time()
    dfnp, res, q_elapsed_time = compute(inputfilename)
    elapsed_time = time.time() - start_time
    print("Training runtime:", q_elapsed_time)
    print("Total runtime:", elapsed_time)

if __name__ == '__main__':
    main()
