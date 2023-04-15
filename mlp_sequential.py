#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:27:28 2023

@author: chakri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:55:02 2023

@author: chakri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
from sklearn.metrics import mean_squared_error
from keras.utils import to_categorical


# set the path to the CSV file
csv_path = "/home/chakri/Chakri/Research/Multi-Layered Perceptron/iris.csv"

# read the CSV file using pandas
df = pd.read_csv(csv_path)

# print the first 5 rows of the dataframe
# print('df: ', df.head())

# remove the last column from the df
df = df.iloc[:, :-1]

# normalizing the dataframe df
for column_name in df.columns:
    # Calculate x_min and x_max for the column
    x_min = df[column_name].min()
    x_max = df[column_name].max()
    
    # Normalize the values in the column and replace them in the DataFrame
    for i, x in enumerate(df[column_name]):
        x_new = (x-x_min)/(x_max-x_min)
        df.at[i, column_name] = x_new
df_norm = df
# print('df_norm: ', df_norm)

class MLP:
    def __init__(self, input_dim, hidden_layers, hidden_dim, output_dim,learningrate):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learningrate = learningrate
        
        # Create empty list to store the weights and biases for each layer
        self.weights = []
        self.biases = []
        self.a = None
        self.a_prev = []
        
        # Add weights and biases for the first-hidden layer
        self.weights.append(np.random.randn(self.input_dim, self.hidden_dim))
        # print("self.weights: ", self.weights)
        self.biases.append(np.zeros((1, self.hidden_dim)))
        # print("self.biases: ", self.biases)
        
        # Add weights and biases for the remianing hidden layers
        for i in range(hidden_layers-1):
            # print("Hidden layer:", i)
            self.weights.append(np.random.randn(self.hidden_dim, self.hidden_dim))
            # print("self.weights: ", self.weights)
            self.biases.append(np.zeros((1, self.hidden_dim)))
            # print("self.biases: ", self.biases)

        
        # Add weights and biases for the output layer
        self.weights.append(np.random.randn(self.hidden_dim, self.output_dim))
        # print("self.weights: ", self.weights)
        self.biases.append(np.zeros((1, self.output_dim)))
        # print("self.biases: ", self.biases)        
        
    def sigmoid(self, x):
        # print("x: ", x)
        # print("1 / (1 + np.exp(-x)) : ", 1 / (1 + np.exp(-x)))        
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # print("")
        # print("---Forward-Pass Calucaltions---")
        # print("")
        # print("X:", X)
        # Loop through each layer and compute the forward pass
        self.a_prev.append(X)
        # print("self.a_prev: ", self.a_prev)
        for i in range(self.hidden_layers+1):
            # print("Layer:", i)
            if i == 0:
                # Input layer
                z = np.dot(X, self.weights[i]) + self.biases[i]
                # print("z:", z)
            else:
                # Hidden layers and output layer
                z = np.dot(self.a, self.weights[i]) + self.biases[i]
                # print("z:", z)

            # Apply activation function
            self.a = self.sigmoid(z)
            # print("self.a:", self.a)
            self.a_prev.append(self.a)
        # print("self.a_prev: ", self.a_prev)
        
        # Return the output
        return self.a, self.a_prev
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)    

    def backward(self, X, y, output, pre_output):
        # print("")
        # print("---Backward-Pass Calucaltions---")
        
        # Create empty list to store the gradients for each layer
        self.weights_gradient = []
        # # print("weights_gradient: ", self.weights_gradient)
        self.biases_gradient = []
        
        # # Loop through each layer and compute the backward pass    
        for i in reversed(range(self.hidden_layers+1)):
            # print("Layer:", i)
            if i == self.hidden_layers:
                # Output layer
                error = y - output
                delta = error * self.sigmoid_derivative(output)
                self.weights_gradient.insert(0, self.learningrate*np.dot(self.a_prev[i-1].T, delta))                
                # print("error: ", error)
                # print("delta_output neurons: ", delta)
                # print("self.weights: ", self.weights)
                # print("self.a_prev: ", self.a_prev)
                # print("self.a_prev: ", self.a_prev[i-1])
                # print("weights_gradient in output layer:", self.weights_gradient)                
            else:
                # Hidden layers and input layer
                error = np.dot(delta, self.weights[i+1].T)
                delta = error * self.sigmoid_derivative(self.a_prev[i+1])
                if i == 0:
                    self.weights_gradient.insert(0, self.learningrate*np.dot(X.T, delta))
                else:
                    self.weights_gradient.insert(0, self.learningrate*np.dot(self.a_prev[i].T, delta))
                # print("error: ", error)
                # print("delta_hidden neurons: ", delta)
                # print("self.a_prev: ", self.a_prev[i])
                # print("self.weights: ", self.weights)
                # print("weights_gradient in hidden layer:", self.weights_gradient)
            self.biases_gradient.insert(0, np.sum(delta, axis=0))
        # print("weights_gradient:", self.weights_gradient)
        # print("self.weights: ", self.weights)
        # print("biases_gradient: ", self.biases_gradient)  
        # print("self.biases: ", self.biases)
        
        for i in range(len(self.weights)):
            self.weights[i] += self.weights_gradient[i]        
        for i in range(len(self.biases)):
            # print("i: ", i)
            # print("self.biases[i]: ", self.biases[i])
            # print("self.biases_gradient[i]: ", self.biases_gradient[i])
            if i == self.hidden_layers: 
                # print("This is a special case")
                bias_i = self.biases[i]  # shape (1, 2)
                bias_gradient_i = np.array(self.biases_gradient[i]).reshape((1, 2))  # shape (1, 2)
                self.biases[i] = bias_i + bias_gradient_i
            else:
                self.biases[i] += self.biases_gradient[i]                
        # print("Updated weights : ", self.weights)
        # print("Updated biases: ", self.biases)      
        # Return the gradients for the weights and biases
        return self.weights, self.biases

    def train(self, X, y, epochs):
        mse_petal_length = []
        mse_petal_width = []
        for i in range(epochs):
            print("EPOCH Count: ", i)
            print("***************************")
            actual_output = []
            diff = []
            for data in range(len(X)):
                # print("Data point:", data)
                # print(data, "th data point:" , X.iloc[data])
                # print("")
                # print("y_train: ", y.iloc[data])
        
                # Forward pass
                output, pre_output = self.forward(X.iloc[data])
                # print("")
                # print("output: ", output)
                # print("")
                # print("pre_output: ", pre_output)
        
                # Backward pass 
                weights, biases = self.backward(np.array(X.iloc[data]).reshape((1, 2)), np.array(y.iloc[data]).reshape((1, 2)), output, pre_output)
                # print("")
                # print("updated weight: ", weights)
                # print("")
                # print("updated biases: ", biases)
                
                # actual_output.append(output)
                y_desired = np.array(y.iloc[data])
                # print("actual_output: ",actual_output)
                # print("y_desired:", y_desired)
                # print("output: ", output)
                diff.append(np.array(y_desired).reshape((1, 2)) - output)
            # print("diff:", diff)


        
            # Compute and print the mean squared error
            
            # print("")
            # print("---Calculating MSE and plotting---")
            # print("")
            diff_arr = np.array(diff).reshape(-1, 2)
            # print("diff_arr: ", diff_arr)
            mean_first = np.mean(diff_arr[:, 0])
            mean_second = np.mean(diff_arr[:, 1])
            mse_petal_length.append(mean_first)
            mse_petal_width.append(mean_second)
           
        plt.plot(range(epochs), mse_petal_length, 'r-', label='petal.length')
        plt.plot(range(epochs), mse_petal_width, 'b-', label='petal.width')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Epoch vs MSE')
        plt.legend()
        plt.show(block=False)


# Load the IRIS dataset
X_train = df.loc[0:120,['sepal.length','sepal.width']]
y_train = df.loc[0:120,['petal.length','petal.width']]
X_test = df.loc[121:149,['sepal.length','sepal.width']]
y_test = df.loc[121:149,['petal.length','petal.width']]

# Define the neural network architecture
mlp = MLP(input_dim=X_train.shape[1], hidden_layers=3,  hidden_dim=5, output_dim=y_train.shape[1], learningrate=0.25)

# # Train the neural network
mlp.train(X_train, y_train, epochs=50)

# # Test the neural network
y_pred, y_pred_all = mlp.forward(X_test)
# print("y_pred: ", y_pred)
# print("y_test: ", y_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
y_test_array = np.array(y_test)
y_test_array = y_test_array.reshape(-1, 2) 
# print("y_test: ", y_test_array)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test_array, axis=1))
print("Accuracy:", accuracy)

