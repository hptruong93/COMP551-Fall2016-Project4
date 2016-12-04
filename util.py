import numpy as np
import sys
import os
import time
from datetime import datetime
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./rnn_data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.input_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def predict(model,X):
    o=model.forward_propagation(X)
    return np.argmax(o, axis=1)

def accuracy(model,X,Y):
    num_words = np.sum([len(y) for y in Y])
    count = 0
    for i,x in enumerate(X):
        y_h = predict(model,x)
        for t,y_t in enumerate(Y[i]):
            if y_t == y_h[t]:
                count = count + 1
    return count*1.0/num_words


def set_model_parameters(model,U=None, V=None, W=None):
    input_dim = model.input_dim
    hidden_dim = model.hidden_dim
    if U==None:
        U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
    if V==None:
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
    if W==None:
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)

def format_input(input_dims,X):
    X_out = []
    for row in X:
        row_out = []
        for col in row:
            row_out.append(format_x_t(input_dims,col))
        X_out.append(row_out) 
    return X_out

def format_x_t(input_dims,x_t):
    input_dim = np.sum(input_dims)
    x_t_final=[0.0]*input_dim
    start = 0
    for i,dim in enumerate(input_dims):
        if dim == 1:
            x_t_final[start] = x_t[i]+0.0
        else:
            x_t_final[start+x_t[i]] = 1.0
        start = start + dim
    return x_t_final
