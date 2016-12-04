import numpy as np
from util import *
from rnn_theano_modified import RNNTheano_modified

_HIDDEN_DIM = 10
_NEPOCH = 100
_LEARNING_RATE = 0.05

#[1 out of 5 labelled positions, env_1 with 2 labels, numerical env_2]
input_dims = [5,2,1]

#10 series of X
#[[0,0,2],[1,1,1],[2,1,3],[3,0,5]]: each list is a timestamp, there are 5 timestamps here
#[0,0,2] : [position with label 0, env_1 with label 0, env_2 = 2]
#all value should be int
X_train = [[[0,0,0.2],[1,1,0.1],[2,1,0.3],[3,0,0.5]],
    [[2,1,0.2],[3,0,0.4]],
    [[1,1,0.2],[2,1,0.4],[3,1,0.9]],
    [[1,0,0.4]],
    [[0,0,0.7],[1,0,0.8]],
    [[3,1,0.2]],
    [[2,0,0.1],[3,1,0.4]],
    [[1,0,0.4],[2,0,0.7],[3,1,0.5]],
    [[0,1,0.5],[1,1,0.9],[2,0,0.4],[3,1,0.2]],
    [[1,0,0.4],[2,0,0.7]]]

#[1,2,3,4]: list of position labels, at time t, y = position at time t+1, y[t]=x[t+1][0]
Y_train = [[1,2,3,4],
    [3,4],
    [2,3,4],
    [2],
    [1,2],
    [4],
    [3,4],
    [2,3,4],
    [1,2,3,4],
    [2,3]]

#change to 1-hot input
X_tr = format_input(input_dims,X_train)

model = RNNTheano_modified(np.sum(input_dims), hidden_dim=_HIDDEN_DIM)
train_with_sgd(model, X_tr[1:], Y_train[1:], nepoch=_NEPOCH, learning_rate=_LEARNING_RATE,evaluate_loss_after=5)

predict(model,X_tr[0])
accuracy(model,X_tr,Y_train)

#reset weights
set_model_parameters(model)

X_train = [[[2,0,0.2],[2,1,0.1],[4,1,0.3],[1,0,0.5],[0,0,0.5],[1,0,0.5],[1,0,0.5]],
    [[1,1,0.2],[2,0,0.4]],
    [[0,1,0.2],[0,1,0.4],[0,1,0.9],[0,1,0.9],[0,1,0.9],[0,1,0.9]],
    [[2,0,0.4],[2,0,0.4],[4,0,0.4],[1,0,0.4],[0,0,0.4],[1,0,0.4],[1,0,0.4],[2,0,0.4]],
    [[2,0,0.7],[0,0,0.8],[2,0,0.8],[0,0,0.8]],
    [[3,1,0.2],[2,1,0.2],[0,1,0.2],[2,1,0.2],[0,1,0.2],[2,1,0.2]],
    [[2,1,0.2],[3,1,0.2],[0,1,0.2],[3,1,0.2],[0,1,0.2],[3,1,0.2],[0,1,0.2],[3,1,0.2]],
    [[2,0,0.1],[4,1,0.4],[1,1,0.4]],
    [[4,0,0.4],[2,0,0.7],[1,1,0.5],[3,1,0.5],[4,0,0.4],[2,0,0.7],[1,1,0.5],[3,1,0.5],[4,0,0.4]],
    [[4,0,0.4]]]

#[1,2,3,4]: list of position labels, at time t, y = position at time t+1, y[t]=x[t+1][0]
Y_train = [[2,4,1,0,1,1,2],
    [2,3],
    [0,0,0,0,0,0],
    [2,4,1,0,1,1,2,3],
    [0,2,0,2],
    [2,0,2,0,2,0,2,0],
    [3,0,3,0,3,0,3,0],
    [4,1,0],
    [2,1,3,4,2,1,3,4,2],
    [1]]

#[2,2,4,1,0,1,1,2]
#[1,2,3]
#[0,0,0,0,0,0,0]
#[2,2,4,1,0,1,1,2,3]
#[2,0,2,0,2]
#[3,2,0,2,0,2,0,2,0]
#[2,3,0,3,0,3,0,3,0]
#[2,4,1,0]
#[4,2,1,3,4,2,1,3,4,2]
#[4,1]
X_tr = format_input(input_dims,X_train)
model = RNNTheano_modified(np.sum(input_dims), hidden_dim=25)
train_with_sgd(model, X_tr[1:], Y_train[1:], nepoch=_NEPOCH, learning_rate=_LEARNING_RATE,evaluate_loss_after=5)

predict(model,X_tr[0])
#accuracy(model,X_tr,Y_train)