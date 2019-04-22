### Jeremy Forest ###
### April 18th 2019 ###


### Comments: would need to check if the code if perfect according to a Hofield net ###
### It makes several iterations whithout updating and then after that can still show updates with flipping neurons until we recover the right patter ###
### not sure if this behavior is normal or if there is still a problem in the code ###

### Hasn't been tested with several input / pattern to recover + code would need to be updated for that in the lear_pattern function ###


import numpy as np
import matplotlib.pyplot as plt


#################################
##### Simulation parameters #####
#################################
DT = 1.
TSIM = 30.
epoch = int(TSIM/DT)

N = 16

pattern = [[-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1]]
pattern_height = 4
pattern_width = 4

degraded_input = [[1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1]]




#########################
######### Code ##########
#########################


class HopfieldNetwork():
    def __init__(self, N):
        self.N = N
        assert N == len(pattern[0])
        assert len(pattern[0]) == len(degraded_input[0])

    def W(self):
        #weight matrix
        return np.zeros([self.N, self.N])

    def X(self):
        # neurons state random initialization
        return [np.random.choice([-1,1]) for n in range(self.N)]

    def energy(self, X, W):
        return -0.5*np.dot(np.dot(np.transpose(X), W), X)


def update_state_async(W, X, theta=0):
    # update state asynchronously
    neurons_order = np.random.permutation(N)
    i = np.random.choice(neurons_order) #choose random neuron
    ## would need to rewrite this to have no redraw from the sampling until every number has been drawn once (maybe this is why it does not update every epoch ?
    X[i] = np.dot(W[i], X) - theta
    if X[i] > 0:
        X[i] = 1
    else:
        X[i] = -1
    return X

def learn_pattern(pattern):
    for pattern_nb in range(len(pattern)):
        for i in range(N):
            for j in range(N):
                if i == j:
                    W[i,j] = 0
                else:
                    W[i,j] = pattern[pattern_nb][i] * pattern[pattern_nb][j]  ### to modify according to multiple pattern when storing multiple pattern (add 1/nb_pattern and the sum term)
            W[i,j] = W[i,j]/len(pattern)
            W[j,i] = W[i,j]
    return W

def input_representation(X, epoch, show=False):
    fig, ax = plt.subplots(1, len(pattern), squeeze=False)
    for i in range(len(pattern)):
        X_prime = np.reshape(X, (4, 4))
        ax[i,0].matshow(X_prime, cmap='gray')
        ax[i,0].set_xticks([])
        ax[i,0].set_yticks([])
        if show:
            plt.show()
        else:
            plt.savefig('neuron activation at time' + str(epoch))

def energy_representation(energy, show=False):
    plt.figure()
    plt.plot(range(epoch), energy)
    # plt.axis([0, epoch, 0, np.amax(energy)])
    if show:
        plt.show()
    else:
        plt.savefig('energy.png')
        plt.close()

###############################
### Instantiate the network ###
###############################
net = HopfieldNetwork(N)
W = net.W()
X = net.X()
energy = []

########################
### Run computations ###
########################
learn_pattern(pattern)
# print('learned input') ; input_representation(pattern)
# print('degraded input') ; input_representation(degraded_input[0])


e=0
X = degraded_input[0]
while e < epoch:
    X_new = update_state_async(W, X, theta=0)
    energy.append(net.energy(X_new, W))
    # import pdb; pdb.set_trace()
    input_representation(X_new, e, show=False)
    e+=1
    X = X_new
energy_representation(energy, show=False)
