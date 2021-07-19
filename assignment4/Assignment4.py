import numpy as np
from matplotlib import pyplot as plt

class Readfile:
    def __init__(self, filename):
        with open(filename, encoding='utf8', errors='ignore') as f:
            self.book_data=f.read()
        self.book_chars=list(set(self.book_data))
        self.c2i={c: i for i, c in enumerate(self.book_chars)}
        self.i2c={i: c for c, i in self.c2i.items()}

class RNN:
    def __init__(self, book_data, c2i, i2c, m=100, seq_length=25, eta=0.1, mu=0, sig=0.01, e=0, epsilon=1e-8, epoch=8, smooth_loss=0, n=200):
        self.book_data=book_data
        self.c2i=c2i
        self.i2c=i2c
        self.m=m
        self.K=len(c2i)
        self.seq_length=seq_length

        self.b=np.zeros((self.m, 1))
        self.m_b=np.zeros(self.b.shape)
        self.c=np.zeros((self.K, 1))
        self.m_c=np.zeros(self.c.shape)
        self.U=np.random.normal(mu, sig, (self.m, self.K))
        self.m_U=np.zeros(self.U.shape)
        self.W=np.random.normal(mu, sig, (self.m, self.m))
        self.m_W=np.zeros(self.W.shape)
        self.V=np.random.normal(mu, sig, (self.K, self.m))
        self.m_V=np.zeros(self.V.shape)

        self.h0=np.zeros((self.m, 1))
        self.eta=eta
        self.e=e
        self.epsilon=epsilon
        self.iter=(len(self.book_data)-1)//self.seq_length
        self.epoch=epoch
        self.smooth_loss=smooth_loss
        self.n=n
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def one_hot(self, K, e, book_data, seq_length):
        x_char=book_data[e:e+seq_length]
        # print(x_char)
        y_char=book_data[e+1:e+seq_length+1]
        # print(y_char)
        X, Y=np.zeros((K, seq_length)), np.zeros((K, seq_length))
        for i in range(seq_length):
            x_id, y_id=self.c2i[x_char[i]], self.c2i[y_char[i]]
            X[x_id, i]=1
            Y[y_id, i]=1
        return X, Y
    
    def compute_loss(self, Y, P):
        assert (Y.shape==P.shape)
        assert (Y.shape==(self.K, self.seq_length))
        loss=-np.trace(np.log(Y.T@P))
        # print(loss)
        return loss
    
    def forward(self, X, h, seq_length, W, U, V, b, c):
        H=np.zeros((self.m, seq_length))
        A=np.zeros((self.m, seq_length))
        P=np.zeros((self.K, seq_length))
        h_=h
        for t in range(seq_length):
            a=W@h_+U@X[:, [t]]+b
            h_=np.tanh(a)
            o=V@h_+c
            p=self.softmax(o)
            H[:, [t]]=h_
            A[:, [t]]=a
            P[:, [t]]=p
        return A, H, P
    
    def tanh_derivative(self, x):
        return 1-np.power(np.tanh(x), 2)
    
    def backward(self, X, Y, h, seq_length, W, U, V, b, c):
        A, H, P=self.forward(X, h, seq_length, W, U, V, b, c)
        grad_h=np.zeros(H.shape)
        grad_a=np.zeros(A.shape)

        G=-(Y-P)
        self.grad_c=np.sum(G, axis=-1, keepdims=True)
        self.grad_V=G@(H.T)
       
        grad_h[:, [-1]]=V.T@G[:, [-1]]
        grad_a[:, [-1]]=grad_h[:, [-1]]*self.tanh_derivative(A[:, [-1]])
        for t_ in range(seq_length-2, -1, -1):
            grad_h[:, [t_]]=V.T@G[:, [t_]]+W.T@grad_a[:, [t_+1]]
            grad_a[:, [t_]]=grad_h[:, [t_]]*self.tanh_derivative(A[:, [t_]])
        # print(grad_h)
        self.grad_b=np.sum(grad_a, axis=-1, keepdims=True)

        # grad_W & grad_U
        H_temp=np.zeros(H.shape)
        H_temp[:, [0]]=h
        H_temp[:, 1:]=H[:, :-1]
        self.grad_W=grad_a@(H_temp.T)
        self.grad_U=grad_a@(X.T)

        # clip
        self.grad_c[self.grad_c>5]=5
        self.grad_c[self.grad_c<-5]=-5

        self.grad_b[self.grad_b>5]=5
        self.grad_b[self.grad_b<-5]=-5

        self.grad_U[self.grad_U>5]=5
        self.grad_U[self.grad_U<-5]=-5

        self.grad_V[self.grad_V>5]=5
        self.grad_V[self.grad_V<-5]=-5

        self.grad_W[self.grad_W>5]=5
        self.grad_W[self.grad_W<-5]=-5        

    def relative_error(self, a, b, h=1e-6):
        assert (a.shape==b.shape)
        e=np.max(np.abs(a-b)/np.maximum(h, (np.abs(a)+np.abs(b))))
        return e
    
    def check_gradients(self, K, e, book_data, seq_length, h0, W, U, V, b, c):
        X, Y=self.one_hot(K, e, book_data, seq_length)
        self.backward(X, Y, h0, seq_length, W, U, V, b, c)
        grad_b, grad_c, grad_V, grad_U, grad_W=self.compute_gradients_num(X, Y, h0, seq_length, W, U, V, b, c)
        
        print('check grad_b...')
        e=self.relative_error(self.grad_b, grad_b)
        print('error for grad_b=', e)

        print('check grad_c...')
        e=self.relative_error(self.grad_c, grad_c)
        print('error for grad_c=', e)

        print('check grad_V...')
        # print('self.grad_V', self.grad_V[:5])
        # print('grad_V', grad_V[:5])
        e=self.relative_error(self.grad_V, grad_V)
        print('error for grad_V=', e)

        print('check grad_U...')
        e=self.relative_error(self.grad_U, grad_U)
        print('error for grad_U=', e)

        print('check grad_W...')
        e=self.relative_error(self.grad_W, grad_W)
        print('error for grad_W=', e)
    
    def ada_grad(self, para, m, g, eta, epsilon):
        m=m+np.power(g, 2)
        para=para-eta*g/(np.sqrt(m+epsilon))
        return m, para
    
    def train(self, plotting=True):
        # check gradients
        # self.check_gradients(self.K, self.e, self.book_data, self.seq_length, self.h0, self.W, self.U, self.V, self.b, self.c)
        
        loss_lst=list()

        for epoch in range(self.epoch):
            self.hprev=self.h0
            self.e=0
            for iter in range(self.iter):
                X, Y=self.one_hot(self.K, self.e, self.book_data, self.seq_length)
                A, H, P=self.forward(X, self.hprev, self.seq_length, self.W, self.U, self.V, self.b, self.c)
                self.backward(X, Y, self.hprev, self.seq_length, self.W, self.U, self.V, self.b, self.c)
                self.m_W, self.W=self.ada_grad(self.W, self.m_W, self.grad_W, self.eta, self.epsilon)
                self.m_U, self.U=self.ada_grad(self.U, self.m_U, self.grad_U, self.eta, self.epsilon)
                self.m_V, self.V=self.ada_grad(self.V, self.m_V, self.grad_V, self.eta, self.epsilon)
                self.m_b, self.b=self.ada_grad(self.b, self.m_b, self.grad_b, self.eta, self.epsilon)
                self.m_c, self.c=self.ada_grad(self.c, self.m_c, self.grad_c, self.eta, self.epsilon)

                self.e+=self.seq_length
                self.hprev=H[:, [-1]]

                loss=self.compute_loss(Y, P)
                if self.smooth_loss==0:
                    self.smooth_loss=loss
                else:
                    self.smooth_loss=0.999*self.smooth_loss+0.001*loss

                if (iter+epoch*self.iter)%100==0:
                    loss_lst.append(self.smooth_loss)
                if (iter+epoch*self.iter)%1000==0:
                    print('iter', iter+epoch*self.iter, 'smooth_loss', self.smooth_loss)
                if (iter+epoch*self.iter)%10000==0:
                    Y_gen=self.generate(X[:, [0]], self.K, self.hprev, 1, self.n, self.W, self.U, self.V, self.b, self.c)
                    s=self.label_to_char(Y_gen, self.i2c)
                    print(s)
                    print()
                    with open('Text_1.txt', 'a', encoding='utf8') as f:
                        f.write('iter: '+str(iter+epoch*self.iter)+'\tsmooth_loss: '+str(self.smooth_loss)+'\n')
                        f.write(s+'\n\n')

        Y_gen=self.generate(X[:, [0]], self.K, self.hprev, 1, 1000, self.W, self.U, self.V, self.b, self.c)
        s=self.label_to_char(Y_gen, self.i2c)
        print(s)
        with open('Text_1.txt', 'a', encoding='utf8') as f:
            f.write('Final model:\n')
            f.write('smooth_loss: '+str(self.smooth_loss)+'\n')
            f.write(s+'\n')

        if plotting:
            plt.plot(loss_lst)
            plt.title('Smooth Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Smooth Loss')
            plt.grid()
            plt.savefig('Loss_1.png')
            plt.show()

    def generate(self, x, K, h, seq_length, n, W, U, V, b, c):
        Y=np.zeros((K, n))
        h_=np.copy(h)
        for i in range(n):
            A, H, P=self.forward(x, h_, seq_length, W, U, V, b, c)
            idx=np.random.choice(K, p=P.flatten())
            Y[idx, i]=1
            h_=H
            x=np.zeros(x.shape)
            x[idx]=1
        return Y

    def label_to_char(self, Y, i2c):
        s=""
        for i in range(Y.shape[1]):
            idx=np.where(Y[:, i]==1)
            assert (len(idx)==1)
            idx=idx[0][0]
            char=i2c[idx]
            s+=char
        return s

    def compute_gradients_num(self, X, Y, h0, seq_length, W, U, V, b, c, h=1e-4):
        grad_V=np.zeros(V.shape)
        grad_W=np.zeros(W.shape)
        grad_U=np.zeros(U.shape)
        grad_b=np.zeros(b.shape)
        grad_c=np.zeros(c.shape)

        # grad_b
        print('calculating grad_b...')
        for i in range(grad_b.shape[0]):
            for j in range(grad_b.shape[1]):
                b_try=np.copy(b)
                b_try[i, j]-=h
                P=self.forward(X, h0, seq_length, W, U, V, b_try, c)[2]
                c1=self.compute_loss(Y, P)

                b_try=np.copy(b)
                b_try[i, j]+=h
                P=self.forward(X, h0, seq_length, W, U, V, b_try, c)[2]
                c2=self.compute_loss(Y, P)
                
                grad_b[i, j]=(c2-c1)/(2*h)
        
        # grad_c
        print('calculating grad_c...')
        for i in range(grad_c.shape[0]):
            for j in range(grad_c.shape[1]):
                c_try=np.copy(c)
                c_try[i, j]-=h
                P=self.forward(X, h0, seq_length, W, U, V, b, c_try)[2]
                c1=self.compute_loss(Y, P)

                c_try=np.copy(c)
                c_try[i, j]+=h
                P=self.forward(X, h0, seq_length, W, U, V, b, c_try)[2]
                c2=self.compute_loss(Y, P)
                
                grad_c[i, j]=(c2-c1)/(2*h)
        
        # grad_V
        print('calculating grad_V...')
        for i in range(grad_V.shape[0]):
            for j in range(grad_V.shape[1]):
                V_try=np.copy(V)
                V_try[i, j]-=h
                P=self.forward(X, h0, seq_length, W, U, V_try, b, c)[2]
                c1=self.compute_loss(Y, P)

                V_try=np.copy(V)
                V_try[i, j]+=h
                P=self.forward(X, h0, seq_length, W, U, V_try, b, c)[2]
                c2=self.compute_loss(Y, P)
                
                grad_V[i, j]=(c2-c1)/(2*h)
        
        # grad_U
        print('calculating grad_U...')
        for i in range(grad_U.shape[0]):
            for j in range(grad_U.shape[1]):
                U_try=np.copy(U)
                U_try[i, j]-=h
                P=self.forward(X, h0, seq_length, W, U_try, V, b, c)[2]
                c1=self.compute_loss(Y, P)

                U_try=np.copy(U)
                U_try[i, j]+=h
                P=self.forward(X, h0, seq_length, W, U_try, V, b, c)[2]
                c2=self.compute_loss(Y, P)

                grad_U[i, j]=(c2-c1)/(2*h)
        
        # grad_W
        print('calculating grad_W...')
        for i in range(grad_W.shape[0]):
            for j in range(grad_W.shape[1]):
                W_try=np.copy(W)
                W_try[i, j]-=h
                P=self.forward(X, h0, seq_length, W_try, U, V, b, c)[2]
                c1=self.compute_loss(Y, P)

                W_try=np.copy(W)
                W_try[i, j]+=h
                P=self.forward(X, h0, seq_length, W_try, U, V, b, c)[2]
                c2=self.compute_loss(Y, P)

                grad_W[i, j]=(c2-c1)/(2*h)
        
        return grad_b, grad_c, grad_V, grad_U, grad_W

def main():
    readfile=Readfile('goblet_book.txt')
    book_data=readfile.book_data
    c2i=readfile.c2i
    i2c=readfile.i2c
    rnn=RNN(book_data, c2i, i2c)
    rnn.train()

if __name__ == '__main__':
    main()