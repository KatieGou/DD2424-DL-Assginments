from functions import *
import numpy.matlib
import matplotlib.pyplot as plt
import math
# np.random.seed(0)

# dropout

A=LoadBatch('data_batch_1')
B=LoadBatch('data_batch_2')
C=LoadBatch('data_batch_3')
D=LoadBatch('data_batch_4')
E=LoadBatch('data_batch_5') # last 1000 as validation
F=LoadBatch('test_batch')

# data=(A[b'data']/255).T
data=np.hstack(((A[b'data']/255).T, (B[b'data']/255).T, (C[b'data']/255).T, (D[b'data']/255).T))
data_E=(E[b'data']/255).T[:, :9000]
data=np.hstack((data, data_E))

labels=A[b'labels']+B[b'labels']+C[b'labels']+D[b'labels']+E[b'labels'][:9000]
labels=np.array(labels)

validation_data=(E[b'data']/255).T[:, 1000:]
validation_labels=E[b'labels'][1000:]
validation_labels=np.array(validation_labels)

test_data=(F[b'data']/255).T
test_labels=F[b'labels']
test_labels=np.array(test_labels)

K, n, d, m=10, data.shape[1], data.shape[0], 300

data_mean=np.mean(data, axis=1)
data=data-numpy.matlib.repmat(data_mean.reshape(-1, 1), 1, data.shape[1])
data_std=np.std(data, axis=1)
data=data/numpy.matlib.repmat(data_std.reshape(-1, 1), 1, data.shape[1])

validation_data=validation_data-numpy.matlib.repmat(data_mean.reshape(-1, 1), 1, validation_data.shape[1])
validation_data=validation_data/numpy.matlib.repmat(data_std.reshape(-1, 1), 1, validation_data.shape[1])

test_data=test_data-numpy.matlib.repmat(data_mean.reshape(-1, 1), 1, test_data.shape[1])
test_data=test_data/numpy.matlib.repmat(data_std.reshape(-1, 1), 1, test_data.shape[1])

def initialize_paras(mu1=0, sigma1=1/math.sqrt(d), mu2=0, sigma2=1/math.sqrt(m)):
    W1=np.random.normal(mu1, sigma1, (m, d))
    b1=np.random.normal(mu1, sigma1, (m, 1))
    W2=np.random.normal(mu2, sigma2, (K, m))
    b2=np.random.normal(mu2, sigma2, (K, 1))
    return W1, b1, W2, b2

def EvaluateClassifier(X, W1, b1, W2, b2):
    s1=W1@X+b1
    h=np.maximum(0, s1)
    h=drop_out_data(h)
    s=W2@h+b2
    p=softmax(s)
    return h, p

def one_hot(y):
    out_Y=np.zeros((K, len(y)))
    for i in range(len(y)):
        out_Y[y[i]][i]=1
    return out_Y

def ComputeCrossEntropy(X, y, W1, b1, W2, b2):
    p=EvaluateClassifier(X, W1, b1, W2, b2)[1]
    Y=one_hot(y)
    loss=0
    for i in range(Y.shape[1]):
        loss+=-math.log(Y.T[i, :]@p[:, i])
    return loss

def ComputeCost(X, y, W1, b1, W2, b2, lamda):
    loss=ComputeCrossEntropy(X, y, W1, b1, W2, b2)
    c=loss/X.shape[1]+lamda*(np.sum(np.square(W1))+np.sum(np.square(W2)))
    return c

def ComputeAccuracy(X, y, W1, b1, W2, b2):
    p=EvaluateClassifier(X, W1, b1, W2, b2)[1]
    largest_idx=p.argmax(axis=0)
    correct_no=0
    for i in range(len(largest_idx)):
        if largest_idx[i]==y[i]:
            correct_no+=1
    acc=correct_no/len(largest_idx)
    return acc

def ComputeGradients(X, y, W1, b1, W2, b2, lamda):
    entries=X.shape[1]
    Y=one_hot(y)
    h, p=EvaluateClassifier(X, W1, b1, W2, b2)
    G=-(Y-p)
    grad_W2=1/entries*G@h.T+2*lamda*W2
    grad_b2=np.mean(G, axis=1).reshape(-1, 1)
    G=W2.T@G
    h[h>0]=1
    G=G*h
    grad_W1=1/entries*G@X.T+2*lamda*W1
    grad_b1=np.mean(G, axis=1).reshape(-1, 1)
    return grad_W1, grad_b1, grad_W2, grad_b2

def ComputeGradientsSlow(X, y, W1, b1, W2, b2, lamda, h=1e-6):
    grad_W1=np.zeros(W1.shape)
    grad_b1=np.zeros(b1.shape)
    grad_W2=np.zeros(W2.shape)
    grad_b2=np.zeros(b2.shape)

    b1_try=np.copy(b1)
    for i in range(b1.shape[0]):
        b1_try[i]=b1_try[i]-h
        c1=ComputeCost(X, y, W1, b1_try, W2, b2, lamda)
        b1_try[i]=b1_try[i]+2*h
        c2=ComputeCost(X, y, W1, b1_try, W2, b2, lamda)
        grad_b1[i]=(c2-c1)/(2*h)

    w1_try=np.copy(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w1_try[i, j]=w1_try[i, j]-h
            c3=ComputeCost(X, y, w1_try, b1, W2, b2, lamda)
            w1_try[i, j]=w1_try[i, j]+2*h
            c4=ComputeCost(X, y, w1_try, b1, W2, b2, lamda)
            grad_W1[i, j]=(c4-c3)/(2*h)
    
    b2_try=np.copy(b2)
    for i in range(b2.shape[0]):
        b2_try[i]=b2_try[i]-h
        c1=ComputeCost(X, y, W1, b1, W2, b2_try, lamda)
        b2_try[i]=b2_try[i]+2*h
        c2=ComputeCost(X, y, W1, b1, W2, b2_try, lamda)
        grad_b2[i]=(c2-c1)/(2*h)
    
    w2_try=np.copy(W2)
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            w2_try[i, j]=w2_try[i, j]-h
            c3=ComputeCost(X, y, W1, b1, w2_try, b2, lamda)
            w2_try[i, j]=w2_try[i, j]+2*h
            c4=ComputeCost(X, y, W1, b1, w2_try, b2, lamda)
            grad_W2[i, j]=(c4-c3)/(2*h)

    return grad_W1, grad_b1, grad_W2, grad_b2

def MiniBatchGD(X, y, n_batch, W1, b1, W2, b2, n_s, n_cycles, eta_min=1e-5, eta_max=1e-1):
    X=drop_out_data(X)
    W1_lst=list()
    b1_lst=list()
    W2_lst=list()
    b2_lst=list()
    cost_lst=list()
    acc_lst=list()
    lamda=coarse_search_lamda(-4, -3, 1.907e-3)
    # print(lamda, end=' ')
    n=X.shape[1]
    n_epochs=2*n_s*n_cycles/(n/n_batch)
    for i in range(int(n_epochs)):
        eta=cyclical_learning_rate(eta_min, eta_max, n_s, i*(n/n_batch))
        for j in range(int(n/n_batch)):
            X_batch=X[:, j*n_batch:(j+1)*n_batch]
            y_batch=y[j*n_batch:(j+1)*n_batch]
            grad_W1, grad_b1, grad_W2, grad_b2=ComputeGradients(X_batch, y_batch, W1, b1, W2, b2, lamda)
            W1=W1-eta*grad_W1
            b1=b1-eta*grad_b1
            W2=W2-eta*grad_W2
            b2=b2-eta*grad_b2
        W1_lst.append(W1)
        b1_lst.append(b1)
        W2_lst.append(W2)
        b2_lst.append(b2)
        cost_lst.append(ComputeCost(X, y, W1, b1, W2, b2, lamda))
        acc_lst.append(ComputeAccuracy(X, y, W1, b1, W2, b2))
    return W1_lst, b1_lst, W2_lst, b2_lst, cost_lst, acc_lst, lamda

def drop_out_data(input_, P=0.9): # modify X in MiniBatchGD, modify h in EvaluateClassifier
    # P: prob of 1
    u=np.array(np.random.choice([0, 1], size=(input_.shape[1], ), p=[1-P, P]))
    return input_*u

def cyclical_learning_rate(eta_min, eta_max, n_s, t):
    cur_t=t-2*n_s*(t//(2*n_s))
    if cur_t<n_s:
        eta=eta_min+cur_t*(eta_max-eta_min)/n_s
    else:
        eta=eta_max-(cur_t-n_s)*(eta_max-eta_min)/n_s
    return eta

def coarse_search_lamda(l_min, l_max, best_lamba):
    # l=l_min+(l_max-l_min)*np.random.rand()
    # if best_lamba==None:
    #     lamda=10**l
    #     return lamda
    # else:
    #     lamda=best_lamba+np.random.choice([-1, 1], p=[0.5, 0.5])*(10**l)
    #     return lamda
    return 0.002

def relative_error(a, n):
    e=np.max(np.abs(a-n)/np.maximum(1e-8, (np.abs(a)+np.abs(n))))
    return e

W1, b1, W2, b2=initialize_paras()
n_batch=100
n_s, n_cycles=2700, 3
# grad_W1, grad_b1, grad_W2, grad_b2=ComputeGradients(data[:, :2], labels[:2], W1, b1, W2, b2, lamda)
# grad_W1_slow, grad_b1_slow, grad_W2_slow, grad_b2_slow=ComputeGradientsSlow(data[:, :2], labels[:2], W1, b1, W2, b2, lamda)

# for iteration in range(10):
W1_lst, b1_lst, W2_lst, b2_lst, cost_lst_training, acc_lst_training, lamda = MiniBatchGD(data, labels, n_batch, W1, b1, W2, b2, n_s, n_cycles)

cost_lst_validation=list()
cost_lst_test=list()

acc_lst_validation=list()
acc_lst_test=list()

for i in range(len(W1_lst)):
    cost_lst_validation.append(ComputeCost(validation_data, validation_labels, W1_lst[i], b1_lst[i], W2_lst[i], b2_lst[i], lamda))
    cost_lst_test.append(ComputeCost(test_data, test_labels, W1_lst[i], b1_lst[i], W2_lst[i], b2_lst[i], lamda))
    acc_lst_validation.append(ComputeAccuracy(validation_data, validation_labels, W1_lst[i], b1_lst[i], W2_lst[i], b2_lst[i]))
    acc_lst_test.append(ComputeAccuracy(test_data, test_labels, W1_lst[i], b1_lst[i], W2_lst[i], b2_lst[i]))

plt.plot(cost_lst_training, label="training")
plt.plot(cost_lst_validation, label="validation")
plt.plot(cost_lst_test, label="test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss in training, validation and test data")
plt.grid()
plt.show()

plt.plot(acc_lst_training, label="training")
plt.plot(acc_lst_validation, label="validation")
plt.plot(acc_lst_test, label="test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy in training, validation and test data")
plt.grid()
plt.show()

print(max(acc_lst_test))
