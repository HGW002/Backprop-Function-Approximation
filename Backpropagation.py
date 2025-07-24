import numpy as np
import matplotlib.pyplot as plt


###
def logsig(x):
    return 1 / (1 + np.exp(-x))

def purelin(x):
    return x
###


def train_network(learning_rate, S1): # Initialize data ( Weight & Biase )
    np.random.seed(42)
    W1 = np.random.rand(S1, 1) - 0.5
    W2 = np.random.rand(1, S1) - 0.5
    b1 = np.random.rand(S1, 1) - 0.5
    b2 = np.random.rand(1) - 0.5

    mse_threshold = 0.001
    max_iterations = 20
    mse = 0.01
    iteration = 0

    # Prepare for plotting
    plt.figure()

    while mse > mse_threshold and iteration < max_iterations:
        mse = 0
        iteration += 1
        predictions = []

        for P in np.arange(-2, 2.1, 0.1):
            T = 1 + np.sin(np.pi * P / 2)  # Target value

            
            a1 = logsig(np.dot(W1, P) + b1) # Feedforward
            a2 = purelin(np.dot(W2, a1) + b2)

            
            error = T - a2 # Calculate error
            mse += error**2
            predictions.append(a2[0])

            
            dlogsig = np.diag((1 - a1.flatten()) * a1.flatten()) #### Backpropagation
            s2 = -2 * error
            s1 = np.dot(dlogsig, W2.T * s2)

            # Update weights and biases
            W2 -= learning_rate * s2 * a1.T
            W1 -= learning_rate * np.outer(s1, P)
            b2 -= learning_rate * s2.flatten()  
            b1 -= learning_rate * s1

        
        if iteration % 10 == 0:
            plt.plot(np.arange(-2, 2.1, 0.1), predictions, 'g:')

    ###### Final plot #######
    P = np.arange(-2, 2.1, 0.1)
    T = 1 + np.sin(np.pi * P / 2)
    plt.plot(P, T, 'r-', label='Original Function')
    plt.plot(P, predictions, 'b+', label='Approximation')
    plt.title(f'Learning Rate = {learning_rate}, S1 = {S1}, Iterations = {iteration}')
    plt.legend()
    plt.xlabel('P')
    plt.ylabel('Target vs. Output')
    plt.show()

    print(f"Learning Rate: {learning_rate}, S1: {S1}")
    print("Final Weights and Biases:")
    print("W1:", W1)
    print("b1:", b1)
    print("W2:", W2)
    print("b2:", b2)
    print("Iterations:", iteration)
    #########################

for lr in [0.1,0.15, 0.2, 0.4]: ####### Train network for different learning rates and S1 values #####
    for S1 in [2, 10]:
        train_network(lr, S1)
