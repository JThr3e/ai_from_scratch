import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) 
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        # Calculate output error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.W2 += self.a1.T.dot(output_delta) * learning_rate
        self.b2 += np.sum(output_delta, axis= 0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(hidden_delta) * learning_rate
        self.b1 += np.sum(hidden_delta, axis= 0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass and weight update
            self.backward(X, y, output, learning_rate)
            
            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage:
if __name__ == "__main__":
    # Sample XOR dataset (input and labels)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create MLP with 2 input neurons, 4 hidden neurons, 1 output neuron
    mlp = MLP(input_size= 2, hidden_size= 4, output_size= 1)
    
    # Train for 10000 epochs with learning rate 0.1
    mlp.train(X, y, epochs= 10000, learning_rate= 0.1)
    
    # Test predictions
    print("\nFinal predictions:")
    for x in X:
        pred = mlp.forward(x.reshape(1, -1))
        print(f"Input: {x} -> Output: {pred[0][0]:.4f}")
