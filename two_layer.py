import numpy as np
import mnist

class MLP:
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, output_size)
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

        # calculate hidden layer 1 error
        hidden_error1 = np.matmul(output_delta, self.W3.T)
        hidden_delta1 = hidden_error1 * self.sigmoid_derivative(self.a2)
        
        # Calculate hidden layer error
        hidden_error = np.matmul(hidden_delta1, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # calculate the gradients at each of the leaves
        W3 = np.matmul(self.a2.T, output_delta)
        b3 = output_delta         
        W2 = np.matmul(self.a1.T, hidden_delta1)
        b2 = hidden_delta1
        W1 = np.matmul(np.expand_dims(x,1), hidden_delta) 
        b1 = hidden_delta

        return [W3, b3, W2, b2, W1, b1] 


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

                params = [self.W3, self.b3, self.W2, self.b2, self.W1, self.b1]
                self.optimizer(params, grads, learning_rate)
            
                # Print loss every 1000 epochs
                if epoch % 1 == 0:
                    loss_sum += np.sum(np.square(correct - output))
            if epoch % 1 == 0:
                loss = loss_sum / float(y.size)
                print(f"Epoch {epoch}, Loss: {loss}")

def print_mnist(x):
    for i in range(0, 28):
        for j in range(0, 28):
            print("X" if x[(i*28)+j] > 100 else " ", end="")
        print()

# Example usage:
if __name__ == "__main__":
    # Sample XOR dataset (input and labels)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    x_train, t_train, x_test, t_test = mnist.load()
    print(x_train.shape)
    print(t_train.shape)

    t_train_proc = np.zeros((60000, 10))
    for i,t in enumerate(t_train):
        t_train_proc[i][t] = 1.0

    print(t_train_proc[0])
    
    # Create MLP with 2 input neurons, 4 hidden neurons, 1 output neuron
    mlp = MLP(input_size= 784, hidden_size=256, hidden_size2=256, output_size= 10)
    
    # Train for 10000 epochs with learning t_train_procrate 0.1
    mlp.train(x_train, t_train_proc, epochs= 100, learning_rate= 0.1)
    
    # Test predictions
    print("\nFinal predictions:")
    for x in x_test[0:10]:
        pred_raw = mlp.forward(x)
        print_mnist(x)
        pred = np.argmax(pred_raw)
        print(f" -> Output: {pred}")
