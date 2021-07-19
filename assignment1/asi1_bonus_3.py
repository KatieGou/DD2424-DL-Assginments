from functions import *
import numpy.matlib
import matplotlib.pyplot as plt
import math

# Xavier

A=LoadBatch('data_batch_1')
B=LoadBatch('data_batch_2')
C=LoadBatch('test_batch')

data=(A[b'data']/255).T
labels=A[b'labels']
labels=np.array(labels)

validation_data=(B[b'data']/255).T
validation_labels=B[b'labels']
validation_labels=np.array(validation_labels)

test_data=(C[b'data']/255).T
test_labels=C[b'labels']
test_labels=np.array(test_labels)

K, n, d=10, 10000, 3072

data_mean=np.mean(data, axis=1)
data=data-numpy.matlib.repmat(data_mean.reshape(-1, 1), 1, data.shape[1])
data_std=np.std(data, axis=1)
data=data/numpy.matlib.repmat(data_std.reshape(-1, 1), 1, data.shape[1])

validation_data=validation_data-numpy.matlib.repmat(data_mean.reshape(-1, 1), 1, validation_data.shape[1])
validation_data=validation_data/numpy.matlib.repmat(data_std.reshape(-1, 1), 1, validation_data.shape[1])

test_data=test_data-numpy.matlib.repmat(data_mean.reshape(-1, 1), 1, test_data.shape[1])
test_data=test_data/numpy.matlib.repmat(data_std.reshape(-1, 1), 1, test_data.shape[1])

def initialize_paras(mu=0, sigma=math.sqrt(1/d)):
    W=np.random.normal(mu, sigma, (K, d))
    b=np.random.normal(mu, sigma, (K, 1))
    return W, b

def EvaluateClassifier(X, W, b):
    s=W@X+b
    p=softmax(s)
    return p

def one_hot(y):
    out_Y=np.zeros((K, len(y)))
    for i in range(len(y)):
        out_Y[y[i]][i]=1
    return out_Y

def ComputeCrossEntropy(X, y, W, b):
    p=EvaluateClassifier(X, W, b)
    Y=one_hot(y)
    l=-np.log(Y.T@p)
    return np.trace(l)

def ComputeCost(X, y, W, b, lamda): # why sum up diagonal?
    loss=ComputeCrossEntropy(X, y, W, b)
    c=loss/X.shape[1]+lamda*np.sum(np.square(W))
    return c

def ComputeAccuracy(X, y, W, b):
    p=EvaluateClassifier(X, W, b)
    largest_idx=p.argmax(axis=0)
    correct_no=0
    for i in range(len(largest_idx)):
        if largest_idx[i]==y[i]:
            correct_no+=1
    acc=correct_no/len(largest_idx)
    return acc

def ComputeGradients(X, y, W, b, lamda):
    entries=X.shape[1]
    Y=one_hot(y)
    p=EvaluateClassifier(X, W, b)
    g=-(Y-p).T
    grad_W=g.T@X.T
    grad_b=np.mean(g, axis=0).reshape(-1, 1)
    grad_W/=entries
    grad_W+=2*lamda*W
    return grad_W, grad_b

def MiniBatchGD(X, y, n_batch, n_epochs, eta, W, b, lamda):
    W_lst=list()
    b_lst=list()
    cost_lst=list()
    acc_lst=list()
    for i in range(n_epochs):
        for j in range(int(X.shape[1]/n_batch)):
            X_batch=X[:, j*n_batch:(j+1)*n_batch]
            y_batch=y[j*n_batch:(j+1)*n_batch]
            grad_W, grad_b=ComputeGradients(X_batch, y_batch, W, b, lamda)
            W=W-eta*grad_W
            b=b-eta*grad_b
        W_lst.append(W)
        b_lst.append(b)
        cost_lst.append(ComputeCost(X, y, W, b, lamda))
        acc_lst.append(ComputeAccuracy(X, y, W, b))
    return W_lst, b_lst, cost_lst, acc_lst

def randomize_data(X, y):
    no_data=X.shape[1]
    idx_lst=list(range(no_data))
    np.random.shuffle(idx_lst)
    X=X[:, idx_lst]
    y=y[idx_lst]
    return X, y

def relative_error(a, n):
    e=np.max(np.abs(a-n)/np.maximum(1e-8, (np.abs(a)+np.abs(n))))
    return e

W, b=initialize_paras()

n_batch, n_epochs, eta, lamda=100, 40, 0.001, 1
W_lst, b_lst, cost_lst_training, acc_lst_training = MiniBatchGD(data, labels, n_batch, n_epochs, eta, W, b, lamda)

cost_lst_validation=list()
cost_lst_test=list()

acc_lst_validation=list()
acc_lst_test=list()

for i in range(len(W_lst)):
    cost_lst_validation.append(ComputeCost(validation_data, validation_labels, W_lst[i], b_lst[i], lamda))
    cost_lst_test.append(ComputeCost(test_data, test_labels, W_lst[i], b_lst[i], lamda))
    acc_lst_validation.append(ComputeAccuracy(validation_data, validation_labels, W_lst[i], b_lst[i]))
    acc_lst_test.append(ComputeAccuracy(test_data, test_labels, W_lst[i], b_lst[i]))

plt.plot(cost_lst_training, label="training")
plt.plot(cost_lst_validation, label="validation")
plt.plot(cost_lst_test, label="test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Loss")
plt.grid()
plt.show()

plt.plot(acc_lst_training, label="training")
plt.plot(acc_lst_validation, label="validation")
plt.plot(acc_lst_test, label="test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Accuracy")
plt.grid()
plt.show()

print("final test acc=", max(acc_lst_test))

montage(W_lst[-1])