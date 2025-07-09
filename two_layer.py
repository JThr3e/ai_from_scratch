import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, hidden_size)
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = np.random.randn(hidden_size, output_size)
        self.b3 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Input to hidden layer
        self.z1 = np.matmul(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Hidden to Hidden
        self.z2 = np.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        # Hidden to output layer
        self.z3 = np.matmul(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3
    
    def backward(self, x, y, output):
        # Calculate output error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Calculate hidden layer error
        hidden_error = np.matmul(output_delta, self.W3.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        W3 = np.matmul(self.a1.T, output_delta)
        b3 = np.sum(output_delta, axis= 0, keepdims=True)
        W1 = np.matmul(np.expand_dims(x,1), hidden_delta) 
        b1 = np.sum(hidden_delta, axis= 0, keepdims=True) 

        return [W3, b3, W1, b1] 


    def optimizer(self, params, grads, learning_rate):
        for param, grad in zip(params, grads):
            param += grad * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            loss_sum = 0.0
            for data,correct in zip(X,y):
                # Forward pass
                output = self.forward(data)
                
                # Backward pass
                grads = self.backward(data, correct, output)

                params = [self.W3, self.b3, self.W1, self.b1]
                self.optimizer(params, grads, learning_rate)
            
                # Print loss every 1000 epochs
                if epoch % 1000 == 0:
                    loss_sum += np.square(correct - output)
            if epoch % 1000 == 0:
                loss = loss_sum / float(y.size)
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage:
if __name__ == "__main__":
    # Sample XOR dataset (input and labels)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create MLP with 2 input neurons, 4 hidden neurons, 1 output neuron
    mlp = MLP(input_size= 2, hidden_size= 4, output_size= 1)
    
    # Train for 10000 epochs with learning rate 0.1
    mlp.train(X, y, epochs= 100000, learning_rate= 0.1)
    
    # Test predictions
    print("\nFinal predictions:")
    for x in X:
        pred = mlp.forward(x.reshape(1, -1))
        print(f"Input: {x} -> Output: {pred[0][0]:.4f}")
