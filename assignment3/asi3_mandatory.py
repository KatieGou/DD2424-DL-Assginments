import numpy as np
from math import floor
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse
import time

Testfriends = ["gou", "sun", "bin", "yusefi", "bianchi", "maki", "dimitriou"]

class Data:

    def __init__(self,filename,validation_filename):
        self.filename = filename
        self.validation_filename = validation_filename
        
    def process_file(self, filename, validation_filename):

        with open(filename, 'r') as f:
            lines = f.readlines()

        with open(validation_filename, 'r') as f:
            lines_validation = f.readlines()

        validation_indexes = lines_validation[0][:-1].split(' ')
        validation_indexes = list(map(int, validation_indexes))

        dataset = {}
        names = []
        labels = []
        dataset["names_train"] = []
        dataset["labels_train"] = []
        dataset["names_validation"] = []
        dataset["labels_validation"] = []

        all_names = ""

        index = 0

        for line in lines:
            temp = line.replace(',', '').lower().split(' ')
            name = ""
            for i in range(len(temp) - 1):
                if i != 0:
                    name += ' '
                name += temp[i]
                all_names += temp[i]
            temp = temp[-1].replace('\n', '')

            names.append(name)
            labels.append(int(temp))

            if (index + 1) in validation_indexes:
                dataset["names_validation"].append(name)
                dataset["labels_validation"].append(int(temp)-1)
            else:
                dataset["names_train"].append(name)
                dataset["labels_train"].append(int(temp)-1)

            index += 1

        dataset["alphabet"] = ''.join(set(all_names)) + ' '
        dataset["d"] = len(dataset["alphabet"])
        dataset["K"] = len(list(set(labels)))
        dataset["n_len"] = len(max(names, key=len))

        dataset["labels_validation"] = np.array(dataset["labels_validation"])
        dataset["labels_train"] = np.array(dataset["labels_train"])

        return dataset
    
    def OneHotLabels(self, labels, N, K):
        balance = np.zeros(K, dtype=int)

        one_hot_labels = np.zeros((N, K))
        for i in range(len(labels)):
            balance[labels[i]] += 1
            one_hot_labels[i][labels[i]] = 1
        return one_hot_labels, balance

    def create_one_hot(self, dataset):
        one_hot_array = np.zeros(
            (dataset["d"] * dataset["n_len"], len(dataset["names_train"])))

        char_to_int = dict((c, i) for i, c in enumerate(dataset["alphabet"]))
        index = 0
        for name in dataset["names_train"]:
            one_hot = np.zeros((dataset["d"], dataset["n_len"]))
            integer_encoded = [char_to_int[char] for char in name]
            i = 0
            for value in integer_encoded:
                letter = np.zeros((dataset["d"]))
                letter[value] = 1
                one_hot[:, i] = letter
                i += 1
            one_hot_array[:dataset["d"]*dataset["n_len"], index] = one_hot.flatten('F')
            index += 1

        dataset["one_hot_train"] = one_hot_array

        one_hot_array = np.zeros(
            (dataset["d"] * dataset["n_len"], len(dataset["names_validation"])))

        char_to_int = dict((c, i) for i, c in enumerate(dataset["alphabet"]))
        index = 0

        for name in dataset["names_validation"]:
            one_hot = np.zeros((dataset["d"], dataset["n_len"]))
            integer_encoded = [char_to_int[char] for char in name]
            i = 0
            for value in integer_encoded:
                letter = np.zeros((dataset["d"]))
                letter[value] = 1
                one_hot[:, i] = letter
                i += 1
            one_hot_array[:dataset["d"]*dataset["n_len"], index] = one_hot.flatten('F')
            index += 1

        dataset["one_hot_validation"] = one_hot_array

        dataset["names_friends"] = Testfriends
        one_hot_array = np.zeros(
            (dataset["d"] * dataset["n_len"], len(dataset["names_friends"])))
        char_to_int = dict((c, i) for i, c in enumerate(dataset["alphabet"]))
        index = 0

        for name in dataset["names_friends"]:
            one_hot = np.zeros((dataset["d"], dataset["n_len"]))
            integer_encoded = [char_to_int[char] for char in name]
            i = 0
            for value in integer_encoded:
                letter = np.zeros((dataset["d"]))
                letter[value] = 1
                one_hot[:, i] = letter
                i += 1
            one_hot_array[:dataset["d"]*dataset["n_len"], index] = one_hot.flatten('F')
            index += 1

        dataset["one_hot_friends"] = one_hot_array

        dataset["one_hot_label_train"], dataset["balance_train"] = self.OneHotLabels(dataset["labels_train"], len(dataset["labels_train"]), dataset["K"])
        dataset["one_hot_label_validation"], dataset["balance_validation"] = self.OneHotLabels(dataset["labels_validation"], len(dataset["labels_validation"]), dataset["K"])
        dataset["one_hot_label_train"] = dataset["one_hot_label_train"].T
        dataset["one_hot_label_validation"] = dataset["one_hot_label_validation"].T
        return dataset

    def final_dataset(self):
        dataset = self.process_file(self.filename, self.validation_filename)
        dataset = self.create_one_hot(dataset)
        return dataset

class ConvNet:
    def __init__(self, dataset, dimensions=(7, 5, 6, 3), eta=0.01, rho=0.9, batch_size=100, n_ite=6000, n_update=300):
        self.eta = eta
        self.rho = rho
        self.dimensions = dimensions  # dimension = (n1, k1, n2, k2)
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_ite = n_ite
        self.n_update = n_update

        self.n_len = dataset["n_len"]
        self.n_len_1 = (dataset["n_len"] - self.dimensions[1] + 1)
        self.n_len_2 = self.dimensions[2] * \
            (self.n_len_1 - self.dimensions[3] + 1)

        self.F1 = np.zeros(
            (dataset["d"], self.dimensions[1], self.dimensions[0]))
        self.F2 = np.zeros(
            (self.dimensions[0], self.dimensions[3], self.dimensions[2]))
        self.W = np.zeros((self.dataset["K"], self.n_len_2))

        self.F1_momentum = np.zeros(self.F1.shape)
        self.F2_momentum = np.zeros(self.F2.shape)
        self.W_momentum = np.zeros(self.W.shape)

        # np.random.seed(0)
        self.initialization()

    def initialization(self): # He initialization
        mu = 0
        sigma = np.sqrt(2 / self.dataset["d"])

        self.F1 = np.random.normal(mu, sigma, self.F1.shape)
        self.F2 = np.random.normal(mu, sigma, self.F2.shape)
        self.W = np.random.normal(mu, sigma, self.W.shape)

    def sigmoid(self, x):
        r = np.exp(x) / sum(np.exp(x))
        return r
    
    def MFMatrix(self, F, n_len):
        (dd, k, nf) = F.shape
        MF = np.zeros(((n_len - k + 1) * nf, n_len * dd))
        VF = F.reshape((dd * k, nf), order='F').T

        for i in range(n_len - k + 1):
            MF[i * nf:(i + 1) * nf, dd * i:dd * i + dd * k] = VF

        return MF

    def MXMatrixGenerate(self, x_input, d, k):
        n_len = int(x_input.shape[0] / d)

        VX = np.zeros((n_len - k + 1, k * d))

        x_input = x_input.reshape((d, n_len), order='F')

        for i in range(n_len - k + 1):
            VX[i, :] = (x_input[:, i:i + k].reshape((k * d, 1), order='F')).T

        return VX

    def MXMatrix(self, x_input, d, k, nf):
        n_len = int(x_input.shape[0] / d)

        MX = np.zeros(((n_len - k + 1) * nf, k * nf * d))
        VX = np.zeros((n_len - k + 1, k * d))

        x_input = x_input.reshape((d, n_len), order='F')

        for i in range(n_len - k + 1):
            VX[i, :] = (x_input[:, i:i + k].reshape((k * d, 1), order='F')).T

        for i in range(n_len - k + 1):
            for j in range(nf):
                MX[i * nf + j:i * nf + j + 1, j * k * d:j * k * d + k * d] = VX[i, :]

        return MX

    def evaluateClassifier(self, X, MF1, MF2, W):
        dot = MF1@X
        X1 = np.where(dot > 0, dot, 0)
        dot = MF2@X1
        X2 = np.where(dot > 0, dot, 0)
        S = W@X2
        P = self.sigmoid(S)
        assert(P.shape == (self.dataset["K"], X.shape[1]))
        return P

    def computeCost(self, X, Y, MF1, MF2, W):
        loss_sum = 0
        for i in range(X.shape[1]):
            x = np.zeros((X.shape[0], 1))
            y = np.zeros((X.shape[0], 1))
            x = X[:, [i]]
            y = Y[:, [i]]
            loss_sum += self.cross_entropy(x, y, MF1, MF2, W)
        loss_sum /= X.shape[1]
        return loss_sum

    def cross_entropy(self, x, y, MF1, MF2, W):
        l = - np.log((y.T)@self.evaluateClassifier(x, MF1, MF2, W))
        assert(len(l) == 1)
        return l[0]

    def computeAccuracy(self, X, Y):
        MF1 = self.MFMatrix(self.F1, self.n_len)
        MF2 = self.MFMatrix(self.F2, self.n_len_1)
        acc = 0
        for i in range(X.shape[1]):
            P = self.evaluateClassifier(X[:, [i]], MF1, MF2, self.W)
            label = np.argmax(P)
            if label == Y[i]:
                acc += 1
        acc /= X.shape[1]
        return acc

    def testing(self, X):
        MF1 = self.MFMatrix(self.F1, self.n_len)
        MF2 = self.MFMatrix(self.F2, self.n_len_1)
        acc = 0
        labels = ["Arabic", "Chinese","Czech", "Dutch", "English", "French", "German", "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese"]
        real_labels=["Chinese", "Chinese", "Chinese", "Arabic", "Italian", "Japanese", "Greek"]
        names = Testfriends # ["gou", "sun", "bin", "yusefi", "bianchi", "maki", "dimitriou"]
        for i in range(X.shape[1]):
            P = self.evaluateClassifier(X[:, [i]], MF1, MF2, self.W)
            label = np.argmax(P)
            label_=labels[label]
            if label_==real_labels[i]:
                acc+=1
            print(names[i] + " is predicted to be " + labels[label])
        acc/=X.shape[1]
        print("accuracy for testing on friends: ", acc)


    def confusion_matrix(self, X, Y, MF1, MF2):
        P = self.evaluateClassifier(X, MF1, MF2, self.W) # shape: (["K"], X.shape[1])
        P = np.argmax(P, axis=0) # (X.shape[1], )
        T = np.argmax(Y, axis=0) # (X.shape[1], )

        M = np.zeros((self.dataset['K'], self.dataset['K']), dtype=int)

        np.set_printoptions(linewidth=100)

        for i in range(len(P)):
            M[T[i]][P[i]] += 1
        print('Confusion matrix:') # row: real cls, col: predicted cls
        print(M)

    def train(self, X, Y, Xv, Yv, graphics=False):
        n = X.shape[1]

        costsTraining = []
        costsValidation = []

        bestW = np.copy(self.W)
        bestF1 = self.F1
        bestF2 = self.F2

        MF1 = self.MFMatrix(self.F1, self.n_len)
        MF2 = self.MFMatrix(self.F2, self.n_len_1)

        bestVal = self.computeCost(Xv, Yv, MF1, MF2, self.W)[0]
        bestEpoch = 0

        print("Init error: ", bestVal)

        MX = []
        for j in tqdm(range(n)):
            MX.append(scipy.sparse.csr_matrix(self.MXMatrix(X[:, [j]], self.dataset['d'], self.dimensions[1], self.dimensions[0])))

        MX = np.array(MX)

        ite = 0
        stop = False

        min_class = min(self.dataset["balance_train"])

        # n_batch = floor(n / self.batch_size)
        n_batch = floor((min_class * len(self.dataset["balance_train"])) / self.batch_size)
        
        start=time.time()
        while not(stop):
            i = 0

            for x in self.dataset["balance_train"]:
                choices = np.random.randint(i, i + x, size=min_class)
                if i == 0:
                    idx = choices
                else:
                    idx = np.append(idx, choices)

                i += x

            np.random.shuffle(idx)

            for j in range(n_batch):
                if stop:
                    break
                j_start = j * self.batch_size
                j_end = (j + 1) * self.batch_size
                if j == n_batch - 1:
                    j_end = len(idx)
                    # j_end = n

                # idx2 = np.random.choice(np.arange(X.shape[1]), self.batch_size)
                # idx2 = np.random.choice(idx, self.batch_size)
                idx2 = idx[j_start: j_end]
                # idx2 = np.arange(j_start, j_end)
                Xbatch = X[:, idx2]
                Ybatch = Y[:, idx2]

                self.evaluation_and_compute_gradients(
                    Xbatch, Ybatch, MF1, MF2, self.W, MX, idx2)
                self.F1 -= self.F1_momentum
                self.F2 -= self.F2_momentum
                self.W -= self.W_momentum
                MF1 = self.MFMatrix(self.F1, self.n_len)
                MF2 = self.MFMatrix(self.F2, self.n_len_1)

                ite += 1

                if ite % 100 == 0:
                    print("Iteration", ite)

                if ite % self.n_update == 0:
                    val = self.computeCost(Xv, Yv, MF1, MF2, self.W)
                    train=self.computeCost(X, Y, MF1, MF2, self.W)
                    print('val cost', val)
                    if val < bestVal:
                        bestVal = np.copy(val)
                        bestW = np.copy(self.W)
                        bestF1 = self.F1
                        bestF2 = self.F2
                        bestEpoch = np.copy(ite)
                    # self.confusion_matrix(X, Y, MF1, MF2)
                    self.confusion_matrix(Xv, Yv, MF1, MF2)
                    if len(costsValidation)>0:
                        if costsValidation[-1] < val:
                            self.eta *= .5
                    costsValidation.append(val)
                    costsTraining.append(train)
                if ite == self.n_ite:
                    stop = True
        
        print("Final loss: ", val)

        self.F1 = np.copy(bestF1)
        self.F2 = np.copy(bestF2)
        self.W = np.copy(bestW)

        MF1 = self.MFMatrix(self.F1, self.n_len)
        MF2 = self.MFMatrix(self.F2, self.n_len_1)

        print("Best iteration: ", bestEpoch)
        print("Best cost: ", self.computeCost(Xv, Yv, MF1, MF2, self.W))

        if (graphics):
            plt.plot(costsTraining, label="Training cost")
            plt.plot(costsValidation, label="Validation cost")
            plt.xlabel('Epoch number')
            plt.ylabel('Cost')
            plt.title('Cost for the training and validation set over the epochs')
            plt.legend(loc='best')
            plt.grid()
            # plt.savefig("train_val_cost.png")
            plt.show()
        print('training used ', round(time.time()-start, 2), 'second')

    def relative_error(self, a, b, h=1e-5):
        e=np.max(np.abs(a-b)/np.maximum(h, (np.abs(a)+np.abs(b))))
        return e

    def evaluation_and_compute_gradients(self, X, Y, MF1, MF2, W, MX, idx):
        gradF1 = np.zeros((self.F1.shape))
        gradF2 = np.zeros((self.F2.shape))
        gradW = np.zeros((self.W.shape))

        dot = MF1@X
        X1 = np.where(dot > 0, dot, 0)
        dot = MF2@X1
        X2 = np.where(dot > 0, dot, 0)
        S = W@X2
        P = self.sigmoid(S)
        assert(P.shape == (self.dataset["K"], X.shape[1]))

        G = -(Y.T - P.T).T

        gradW = G@X2.T / X2.shape[1]

        G = G.T@W
        S2 = np.where(X2 > 0, 1, 0)
        G = np.multiply(G.T, S2)

        n = X1.shape[1]
        for j in range(n):
            xj = X1[:, [j]]
            gj = G[:, [j]]

            # Mj = self.MXMatrix(
            #      xj, self.dimensions[0], self.dimensions[3], self.dimensions[2]) # n1, k2, n2
            # v = gj.T@Mj
            # gradF2 += v.reshape(self.F2.shape, order='F') / n
            

            MjGen = self.MXMatrixGenerate(xj, self.dimensions[0], self.dimensions[3]) # n1, k2
            a = gj.shape[0]
            gj = gj.reshape((int(a / self.dimensions[2]), self.dimensions[2]))
            v2 = MjGen.T@gj
            gradF2 += v2.reshape(self.F2.shape, order='F') / n
        
        G = G.T@MF2
        S1 = np.where(X1 > 0, 1, 0)
        G = np.multiply(G.T, S1)

        n = X.shape[1]
        for j in range(n):
            gj = G[:, [j]]
            xj = X[:, [j]]

            # Mj = self.MXMatrix(
            #     xj, self.dataset['d'], self.dimensions[1], self.dimensions[0]) # k1, n1

            Mj = np.asarray(MX[idx[j]].todense())

            v = gj.T@Mj
            gradF1 += v.reshape(self.F1.shape, order='F') / n

        self.W_momentum = self.W_momentum * self.rho + self.eta * gradW
        self.F2_momentum = self.F2_momentum * self.rho + self.eta * gradF2
        self.F1_momentum = self.F1_momentum * self.rho + self.eta * gradF1

    def computeGradientsNumerical(self, X, Y, MF1, MF2):
        h = 1e-5

        grad_F1 = np.zeros(self.F1.shape)
        grad_W = np.zeros(self.W.shape)
        grad_F2 = np.zeros(self.F2.shape)

        (a, b, c) = self.F1.shape

        print("Computing F1 grad")

        for i in tqdm(range(c)):
            for j in range(b):
                for k in range(a):
                    F1_try = np.copy(self.F1)
                    F1_try[k, j, i] -= h
                    MF1_try = self.MFMatrix(F1_try, self.n_len)

                    l1 = self.computeCost(X, Y, MF1_try, MF2, self.W)

                    F1_try = np.copy(self.F1)
                    F1_try[k, j, i] += h
                    MF1_try = self.MFMatrix(F1_try, self.n_len)

                    l2 = self.computeCost(X, Y, MF1_try, MF2, self.W)

                    grad_F1[k, j, i] = (l2 - l1) / (2 * h)

        print("Computing F2 grad")

        (a, b, c) = self.F2.shape

        for i in tqdm(range(c)):
            for j in range(b):
                for k in range(a):
                    F2_try = np.copy(self.F2)
                    F2_try[k, j, i] -= h
                    MF2_try = self.MFMatrix(F2_try, self.n_len_1)

                    l1 = self.computeCost(X, Y, MF1, MF2_try, self.W)[0]

                    F2_try = np.copy(self.F2)
                    F2_try[k, j, i] += h
                    MF2_try = self.MFMatrix(F2_try, self.n_len_1)

                    l2 = self.computeCost(X, Y, MF1, MF2_try, self.W)[0]

                    grad_F2[k, j, i] = (l2 - l1) / (2 * h)

        print("Computing W grad")

        for i in tqdm(range(self.W.shape[0])):
            for j in range(self.W.shape[1]):
                W_try = np.copy(self.W)
                W_try[i][j] -= h
                l1 = self.computeCost(X, Y, MF1, MF2, W_try)[0]

                W_try = np.copy(self.W)
                W_try[i][j] += h
                l2 = self.computeCost(X, Y, MF1, MF2, W_try)[0]

                grad_W[i, j] = (l2 - l1) / (2 * h)

        return grad_F1, grad_F2, grad_W

def main():
    filename = "dataset/ascii_names.txt"
    validation_filename = "dataset/Validation_Inds.txt"

    print("Loading dataset...")
    process = Data(filename,validation_filename)
    dataset = process.final_dataset()
    print("Dataset loaded!")

    print(dataset["balance_train"])
    
    conv = ConvNet(dataset)
    conv.train(dataset["one_hot_train"], dataset["one_hot_label_train"], dataset["one_hot_validation"], dataset["one_hot_label_validation"],graphics=True)

    print("Final accuracy:", conv.computeAccuracy(dataset["one_hot_validation"], dataset["labels_validation"]))
    print("Test random names:")
    conv.testing(dataset["one_hot_friends"])

    
if __name__ == "__main__":
    main()
