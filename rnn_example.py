import numpy as np
from util import *
from rnn_theano_modified import RNNTheano_modified

def run_example():
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
    train_with_sgd(model, X_tr[1:], Y_train[1:], nepoch= 100, learning_rate=0.05,evaluate_loss_after=5)

    predict(model,X_tr[0])
    #accuracy(model,X_tr,Y_train)

def CV_accuracy(X_tr,Y_train,nepoch=500):
    correct = 0
    total = 0
    for i in range(len(X_tr)/5):
        set_model_parameters(model)
        X_tr_cv = X_tr[0:i*5] + X_tr[(i+1)*5:]
        Y_tr_cv = Y_train[0:i*5] + Y_train[(i+1)*5:]
        train_with_sgd(model, X_tr_cv, Y_tr_cv, nepoch=100, learning_rate=model.learning_rate,evaluate_loss_after=nepoch-1)
        correct += accuracy(model,X_tr[i*5:(i+1)*5],Y_train[i*5:(i+1)*5])*len(X_tr[i])
        total += len(X_tr[i])
    return correct/float(total)

if __name__ == "__main__":

    from data_reader import *
    birds = get_birds()
    average_birds(birds)
    X_train = []
    for b in birds:
        seq = []
        for e in b['events']:
            seq.append([to_cluster(e),e[1].month])
        X_train.append(seq)

    Y_train = []
    for x in X_train:
        Y_train.append([x_t[0] for x_t in x])

    for x in X_train:
        del x[-1]

    for y in Y_train:
        del y[0]

    input_dims = [11,1]

    del X_train[20]
    del Y_train[20]

    X_tr = format_input(input_dims,X_train)

    model = RNNTheano_modified(np.sum(input_dims), hidden_dim=50)
    train_with_sgd(model, X_tr[10:], Y_train[10:], nepoch=200, learning_rate=model.learning_rate,evaluate_loss_after=5)

    predict(model,X_tr[0])
    accuracy(model,X_tr[10:],Y_train[10:])
    accuracy(model,X_tr[:10],Y_train[:10])

    for i in range(11):
        acc = accuracy_in_month(model,X_tr,Y_train,i)
        print 'month %d: accuracy %f' %(i,acc)