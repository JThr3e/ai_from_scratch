import numpy as np
import mnist
from tqdm import tqdm

import multiprocessing as mp
from multiprocessing import shared_memory

num_processes = 12
barrier = mp.Barrier(num_processes)

class MLP:
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, grad_sync, lock, pid):
        # Initialize weights and biases with random values

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        np.random.seed(42) # ensure random params are same across all replicas
        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))
                  
        self.W2 = np.random.randn(hidden_size, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))
                  
        self.W3 = np.random.randn(hidden_size2, output_size)
        self.b3 = np.zeros((1, output_size))

        self.grad_sync = grad_sync
        self.lock = lock
        self.pid = pid
    
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
        barrier.wait()

        with self.lock:
            self.grad_sync[0] += grads[0] / num_processes
            self.grad_sync[1] += grads[1] / num_processes
            self.grad_sync[2] += grads[2] / num_processes
            self.grad_sync[3] += grads[3] / num_processes
            self.grad_sync[4] += grads[4] / num_processes
            self.grad_sync[5] += grads[5] / num_processes

        barrier.wait()

        with self.lock:
            for param, grad in zip(params, self.grad_sync):
                param += grad * learning_rate

        barrier.wait()

        with self.lock:
            if self.pid == 0:
                self.grad_sync[0][:, :] = 0 
                self.grad_sync[1][:, :] = 0 
                self.grad_sync[2][:, :] = 0 
                self.grad_sync[3][:, :] = 0 
                self.grad_sync[4][:, :] = 0 
                self.grad_sync[5][:, :] = 0 

    def zero_grad(self):
        W3_shape=(self.hidden_size2, self.output_size)
        b3_shape=(1, self.output_size)
        W2_shape=(self.hidden_size, self.hidden_size2)
        b2_shape=(1, self.hidden_size2)
        W1_shape=(self.input_size, self.hidden_size)
        b1_shape=(1, self.hidden_size)
        return [np.zeros(W3_shape), np.zeros(b3_shape), np.zeros(W2_shape), np.zeros(b2_shape), np.zeros(W1_shape), np.zeros(b1_shape)]
    
    def train(self, X, y, epochs, learning_rate):
        mb_size = 400
        for epoch in range(epochs):
            loss_sum = 0.0
            minibatch_grads = self.zero_grad()
            correct_count = 0
            for i,(data,correct) in tqdm(enumerate(zip(X, y))) if self.pid == 0 else enumerate(zip(X, y)):
                # Forward pass
                output = self.forward(data)
                
                # Backward pass
                grads = self.backward(data, correct, output)

                for g, mb in zip(grads, minibatch_grads):
                    mb += g

                if i % mb_size == 0:
                    grads = [minibatch_grads[0], minibatch_grads[1], minibatch_grads[2], minibatch_grads[3], minibatch_grads[4], minibatch_grads[5]]
                    params = [self.W3, self.b3, self.W2, self.b2, self.W1, self.b1]
                    self.optimizer(params, grads, learning_rate)
                    minibatch_grads = self.zero_grad()
            
                # Print loss every epoch
                if np.argmax(correct) == np.argmax(output):
                    correct_count += 1
                loss_sum += np.sum(np.square(correct - output))
            if self.pid == 0:
                loss = loss_sum / float(y.size)
                accuracy = (correct_count / float(y.shape[0])) * 100
                print(f"Epoch {epoch}, Loss: {loss}, Acc: {accuracy}")

def print_mnist(x):
    for i in range(0, 28):
        for j in range(0, 28):
            print("X" if x[(i*28)+j] > 100 else " ", end="")
        print()


def dp_worker(pid, lock, grad_sync_mem, sizes, data, label, epochs, learning):
    
    W3_shm = shared_memory.SharedMemory(name=grad_sync_mem[0])
    b3_shm = shared_memory.SharedMemory(name=grad_sync_mem[1])
    W2_shm = shared_memory.SharedMemory(name=grad_sync_mem[2])
    b2_shm = shared_memory.SharedMemory(name=grad_sync_mem[3])
    W1_shm = shared_memory.SharedMemory(name=grad_sync_mem[4])
    b1_shm = shared_memory.SharedMemory(name=grad_sync_mem[5])
    input_size= sizes[0]
    hidden_size= sizes[1]
    hidden_size2=sizes[2]
    output_size=sizes[3]

    W3_shape=(hidden_size2, output_size)
    b3_shape=(1, output_size)
    W2_shape=(hidden_size, hidden_size2)
    b2_shape=(1, hidden_size2)
    W1_shape=(input_size, hidden_size)
    b1_shape=(1, hidden_size)
    grad_dtype = np.float64

    W3_grad = np.ndarray(W3_shape, dtype=grad_dtype, buffer=W3_shm.buf)
    b3_grad = np.ndarray(b3_shape, dtype=grad_dtype, buffer=b3_shm.buf)
    W2_grad = np.ndarray(W2_shape, dtype=grad_dtype, buffer=W2_shm.buf)
    b2_grad = np.ndarray(b2_shape, dtype=grad_dtype, buffer=b2_shm.buf)
    W1_grad = np.ndarray(W1_shape, dtype=grad_dtype, buffer=W1_shm.buf)
    b1_grad = np.ndarray(b1_shape, dtype=grad_dtype, buffer=b1_shm.buf)

    grad_sync = [W3_grad, b3_grad, W2_grad, b2_grad, W1_grad, b1_grad]

    mlp = MLP(input_size=input_size, hidden_size=hidden_size, hidden_size2=hidden_size2, output_size=output_size, grad_sync=grad_sync, lock=lock, pid=pid)

    # Train for 10000 epochs with learning t_train_procrate 0.1
    mlp.train(data, label, epochs=epochs, learning_rate=learning)


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


    input_size= 784
    hidden_size=512
    hidden_size2=256
    output_size=10
    sizes = [input_size, hidden_size, hidden_size2, output_size]

    grad_dtype = np.float64
    W3_shape=(hidden_size2, output_size)
    b3_shape=(1, output_size)
    W2_shape=(hidden_size, hidden_size2)
    b2_shape=(1, hidden_size2)
    W1_shape=(input_size, hidden_size)
    b1_shape=(1, hidden_size)
    grad_dtype = np.float64



    W3_shm =shared_memory.SharedMemory(create=True, size=np.prod(W3_shape) * np.dtype(grad_dtype).itemsize)
    b3_shm =shared_memory.SharedMemory(create=True, size=np.prod(b3_shape) * np.dtype(grad_dtype).itemsize)
    W2_shm =shared_memory.SharedMemory(create=True, size=np.prod(W2_shape) * np.dtype(grad_dtype).itemsize)
    b2_shm =shared_memory.SharedMemory(create=True, size=np.prod(b2_shape) * np.dtype(grad_dtype).itemsize)
    W1_shm =shared_memory.SharedMemory(create=True, size=np.prod(W1_shape) * np.dtype(grad_dtype).itemsize)
    b1_shm =shared_memory.SharedMemory(create=True, size=np.prod(b1_shape) * np.dtype(grad_dtype).itemsize)

    W3_data = np.ndarray(W3_shape, dtype=grad_dtype, buffer=W3_shm.buf)
    b3_data = np.ndarray(b3_shape, dtype=grad_dtype, buffer=b3_shm.buf)
    W2_data = np.ndarray(W2_shape, dtype=grad_dtype, buffer=W2_shm.buf)
    b2_data = np.ndarray(b2_shape, dtype=grad_dtype, buffer=b2_shm.buf)
    W1_data = np.ndarray(W1_shape, dtype=grad_dtype, buffer=W1_shm.buf)
    b1_data = np.ndarray(b1_shape, dtype=grad_dtype, buffer=b1_shm.buf)

    W3_data[:,:] = 0 
    b3_data[:,:] = 0
    W2_data[:,:] = 0
    b2_data[:,:] = 0
    W1_data[:,:] = 0
    b1_data[:,:] = 0

    grad_sync_mem_names = [W3_shm.name, b3_shm.name, W2_shm.name, b2_shm.name, W1_shm.name, b1_shm.name]
    print(x_train.shape)
    print(t_train_proc.shape)

    lock = mp.Lock()
    processes = []
    for i in range(0, num_processes):
        whole_size = x_train.shape[0]
        size = int(whole_size / num_processes)
        start = i*size
        end = (i+1)*size
        data = x_train[start:end][:]
        labels = t_train_proc[start:end][:]
        print(data.shape)
        print(labels.shape)
        p = mp.Process(target=dp_worker, args=(i, lock, grad_sync_mem_names, sizes, data, labels, 100, 0.01))
        p.start()

    for p in processes:
        p.join()

    #grad_sync = [W3_data, b3_data, W2_data, b2_data, W1_data, b1_data]
    #mlp = MLP(input_size= 784, hidden_size=256, hidden_size2=256, output_size= 10, grad_sync=grad_sync, lock=mp.Lock(), pid=0)
    #mlp.train(x_train, t_train_proc, epochs= 100, learning_rate= 0.01)
    
    # Test predictions
    #print("\nFinal predictions:")
    #for x in x_test[0:10]:
    #    pred_raw = mlp.forward(x)
    #    print_mnist(x)
    #    pred = np.argmax(pred_raw)
    #    print(f" -> Output: {pred}")
