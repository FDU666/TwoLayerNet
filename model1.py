import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def ReLU(x):
    return np.maximum(0, x)

class Two_Layer_Net():
    """
    input layer dim = D; hidden layer dim = H
    input -> fully connected layer -> ReLU -> fully connected layer -> softmax
    """

    def __init__(self, input_size=0, hidden_size=0, output_size=0, std=1e-4):
        """

        W1:  (D, H)
        b1:  (H,)
        W2:  (H, C)
        b2:  (C,)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros((1,hidden_size))
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros((1,output_size))


    def loss(self, X, y, regularization=0.0):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        hidden1 = ReLU(X.dot(W1)+b1)      # hidden layer1
        output = hidden1.dot(W2)+b2       # output layer

        softmax = np.exp(output)  # Softmax (N,C)
        for i in range(0, N):
            softmax[i, :] /= np.sum(softmax[i, :])

        loss = 0  # loss
        data_loss = 0
        for i in range(0, N):
            data_loss += -np.log(softmax[i, y[i]])
        reg_loss = 0.5*regularization*(np.sum(W1*W1)+np.sum(W2*W2))
        loss = data_loss/N + reg_loss

        # back propagation
        gradient = {}
        dl = softmax.copy()
        for i in range(0, N):
            dl[i, y[i]] -= 1
        dl /= N
        gradient['W2'] = hidden1.T.dot(dl) + regularization*W2
        gradient['b2'] = np.sum(dl, axis=0, keepdims=True)
        # ReLU层
        dh1 = dl.dot(W2.T)
        dh1 = (hidden1 > 0)*dh1
        gradient['W1'] = X.T.dot(dh1) + regularization*W1
        gradient['b1'] = np.sum(dh1, axis=0, keepdims=True)

        return loss, gradient

    def train(self, X_train, y_train, X_test, y_test,
              learning_rate=5e-3, lr_decay=0.9,decay_steps = 100,
              regulariaztion=1e-3, iteration=300, batch_size=1000,mu=0.9,mu_increase=1.0,):
    
        """    
        SGD
        Inputs:    
        - X_train:  (N, D)    
        - y_train:(N,) 
        - X_test:  (N_test, D)     
        - y_test:  (N_test,)     
        - learning_rate: 学习率    
        - lr_decay: 学习率衰减因子
        - reg: L2正则化  
        - batch_size:     
        """

        N = X_train.shape[0]
        v_W2, v_b2 = 0.0, 0.0
        v_W1, v_b1 = 0.0, 0.0
        train_loss_history = []
        test_loss_history = []
        test_accuracy_history = []            
        X_batch = None
        y_batch = None

        
        for i in tqdm(range(iteration)):

            sample_index = np.random.choice(N,batch_size,replace =True)
            X_batch = X_train[sample_index]
            y_batch = y_train[sample_index]

            loss, grads = self.loss(X_batch, y_batch, regulariaztion)
            train_loss_history.append(loss)

            # SGD
            v_W2 = mu * v_W2 - learning_rate * grads['W2']
            self.params['W2'] += v_W2
            v_b2 = mu * v_b2 - learning_rate * grads['b2']
            self.params['b2'] += v_b2
            v_W1 = mu * v_W1 - learning_rate * grads['W1']
            self.params['W1'] += v_W1
            v_b1 = mu * v_b1 - learning_rate * grads['b1']
            self.params['b1'] += v_b1

            loss, grads = self.loss(X_test, y_test, regulariaztion)
            test_loss_history.append(loss)
            test_accuracy = (self.predict(X_test) == y_test).mean()
            test_accuracy_history.append(test_accuracy)
            if i % decay_steps ==0:
                learning_rate = learning_rate * lr_decay
                mu *= mu_increase
        return train_loss_history,test_loss_history,test_accuracy_history
    
    def predict(self, X):
        y_pred = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden1 = ReLU(X.dot(W1)+b1)
        output = hidden1.dot(W2)+b2
        y_pred = np.argmax(output, axis=1)

        return y_pred

    def save_model(self, file):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        np.savez(file, W1=W1, b1=b1, W2=W2, b2=b2)

    def load_model(self, file):
        data = np.load(file)
        self.params['W1'] = data['W1']
        self.params['b1'] = data['b1']
        self.params['W2'] = data['W2']
        self.params['b2'] = data['b2']

    def para_model(self):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        return W1, b1, W2, b2

if __name__ == "__main__":

    np.random.seed(123)
    train_images, train_labels = load_mnist('./minist')
    test_images, test_labels = load_mnist('./minist', 't10k')
    # print('Train data shape: ', train_images.shape)
    # print('Train labels shape: ', train_labels.shape)
    # print('Test data shape: ', test_images.shape)
    # print('Test labels shape: ', test_labels.shape)

    twolayermodel = Two_Layer_Net(train_images.shape[1], 100, 10)
    train_loss_history, test_loss_history, test_acc_history = twolayermodel.train(train_images, train_labels, test_images, test_labels)

    # print(test_acc_history)
    W1, b1, W2, b2 = twolayermodel.para_model()
    # print(W1.shape)
    # print(b1.shape)
    # print(W2.shape)
    # print(b2.shape)

    twolayermodel.save_model('./twolayermodel.npz')


    plt.figure(figsize=(8, 3))
    plt.plot(train_loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss history')
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(test_loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Testing Loss')
    plt.title('Testing Loss history')
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(test_acc_history)
    plt.xlabel('Iteration')
    plt.ylabel('Testing Accuracy')
    plt.title('Testing Accuracy history')
    plt.show()



    plt.figure(figsize=(16, 3))
    plt.imshow(W1, cmap='Greys_r')
    plt.show()

    plt.figure(figsize=(16, 3))
    plt.imshow(b1, cmap='Greys_r')
    plt.show()

    plt.figure(figsize=(16, 3))
    plt.imshow(W2, cmap='Greys_r')
    plt.show()

    plt.figure(figsize=(16, 3))
    plt.imshow(b2, cmap='Greys_r')
    plt.show()


