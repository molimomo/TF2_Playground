from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import asarray
from numpy.random import seed
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
from keras.initializers import glorot_uniform
from keras import backend as K
import numpy as np
import os
import argparse
import tensorflow as tf
import random as rn
#os.environ['PYTHONHASHSEED'] = str(0)
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep CNN.")
    parser.add_argument('--option', nargs='?', default='h_rank',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--target_rank', type=int, default=2,
                        help='Target rank for Conv low rank weight matrix.')
    parser.add_argument('--dataset', nargs='?', default='mnist',
                        help='Choose a Dataset.')
    parser.add_argument('--history_path', nargs='?', default='./history/',
                        help='The folder path to save history.')
    parser.add_argument('--model_path', nargs='?', default='./pretrain/',
                        help='The folder path to save trained model.')
    parser.add_argument('--epoch', type=int, default=3,
                        help='Training Epoch.')
    parser.add_argument('--seed_part', type=int, default=0,
                        help='Partition for random seed.')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed.')
    parser.add_argument('--perseed', type=int, default=0,
                        help='seed.')
    parser.add_argument('--channel', type=int, default=1,
                        help='Number of channel.')
    parser.add_argument('--kernel', type=int, default=1,
                        help='Number of kernel.')
    parser.add_argument('--kernel_size',  type=int, default=3,
                        help='The dimension for a kernel.')
    parser.add_argument('--identity_option', type=int, default=1,
                        help='0: unscalaring, 1: scalar to 1')

    return parser.parse_args()


def get_identity_matrix(rows, cols):
    res = np.zeros((rows, cols))
    row_bound = int(rows/2)
    col_bound = int(cols/2)
    for i in range(rows):
        for j in range(cols):
            if (i < row_bound and j < col_bound) or (i >= row_bound and j >= col_bound) :
                res[i, j] = 1
    return res

## The shape of Conv2D weights: [kernel_row, kernel_col, number of channels, number of filters]
def _init_conv_weights(option, num_kernels, kernel_row, kernel_col, num_channels, target_rank,seed):
    RANDOM_STATE = seed
    rn.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    print(option)
    print('Seed: ' + str(seed))
    if option == 'h_rank':
        weights = np.random.normal(size=(kernel_row, kernel_col, num_channels, num_kernels))
    elif option == 'l_rank':
        sub_weights = np.random.normal(size=(kernel_row,target_rank))
        weights = sub_weights.dot(sub_weights.T)
    elif option =='HL':
        h_rank_weights = np.random.normal(size=(kernel_row, kernel_col))
        l_rank_weights = np.random.normal(size=(kernel_row,target_rank))
        l_rank_weights = l_rank_weights.dot(l_rank_weights.T)
        weights = h_rank_weights.dot(l_rank_weights)
    elif option =='l_rank_identity':
        sub_weights = get_identity_matrix(kernel_row, target_rank)
        weights = sub_weights.dot(sub_weights.T)
        if (args.identity_option != 0):
            weights[weights>1] = 1
    elif option == 'HL_identity':
        h_rank_weights = np.random.normal(size=(kernel_row, kernel_col))
        l_rank_weights = get_identity_matrix(kernel_row, target_rank)
        l_rank_weights = l_rank_weights.dot(l_rank_weights.T)
        if (args.identity_option != 0):
            l_rank_weights[l_rank_weights>1] = 1
        weights = h_rank_weights.dot(l_rank_weights)
    elif option == 'dual_low_z_v':
        z = np.random.normal(size=(kernel_row,target_rank))
        v = np.random.normal(size=(kernel_row,target_rank))
        w = z.dot(z.T)
        i = v.dot(v.T)
        weights = w.dot(i)
    elif option == 'dual_low_zi_v':
        z = get_identity_matrix(kernel_row, target_rank)
        v = np.random.normal(size=(kernel_row, target_rank))
        w = z.dot(z.T)
        if (args.identity_option != 0):
            w[w>1] = 1
        i = v.dot(v.T)
        weights = w.dot(i)
    elif option == 'dual_low_z_vi':
        z = np.random.normal(size=(kernel_row, target_rank))
        v = get_identity_matrix(kernel_row, target_rank)
        w = z.dot(z.T)
        i = v.dot(v.T)
        if (args.identity_option != 0):
            i[i>1] = 1
        weights = w.dot(i)
    elif option == 'dual_low_zi_vi':
        z = get_identity_matrix(kernel_row, target_rank)
        v = get_identity_matrix(kernel_row, target_rank)
        w = z.dot(z.T)
        i = v.dot(v.T)
        if (args.identity_option != 0):
            w[w > 1] = 1
            i[i > 1] = 1
        weights = w.dot(i)
    weights = np.reshape(weights, (kernel_row, kernel_col, num_channels, num_kernels))
    return weights

def load_datasets(dataset):
    if dataset == 'mnist':
        return mnist.load_data()
    elif dataset == 'fashion_mnist':
        return fashion_mnist.load_data()
    elif dataset == 'cifar10':
        return cifar10.load_data()
    elif dataset == 'cifar100':
        return cifar100.load_data()

def runCNN(seed, args):
    tf.random.set_seed(seed)
    RANDOM_STATE = seed
    rn.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    #download mnist data and split into train and test sets
    print('Data set: ' + str(args.dataset))
    (X_train, y_train), (X_test, y_test) = load_datasets(args.dataset)
    [num_train, rows, cols] = X_train.shape
    num_test = X_test.shape[0]

    #reshape data to fit model
    X_train = X_train.reshape(num_train,rows,cols, int(args.channel))
    X_test = X_test.reshape(num_test,rows,cols,int(args.channel))
    num_class = len(np.unique(np.concatenate((np.unique(y_train), np.unique(y_test)))))

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    #create model
    model = Sequential()

    #add model layers
    num_kernels = int(args.kernel)
    kernel_size = int(args.kernel_size)
    target_rank = int(args.target_rank)
    num_channel = int(args.channel)
    conv_2d = Conv2D(num_kernels, kernel_size=kernel_size, activation='relu', input_shape=(rows, cols, num_channel))
    model.add(conv_2d)
    weights = _init_conv_weights(args.option, num_kernels, kernel_size, kernel_size, num_channel, target_rank, seed)
    conv_2d.set_weights([weights, asarray([0.0])])
    model.add(Flatten())
    model.add(Dense(num_class, activation='softmax'))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    res = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=args.epoch)
    model_str = "_".join([args.option, "kernel_size", str(args.kernel_size),
                          "num_kernel", str(args.kernel),
                          "channel", str(args.channel),
                          "rank",str(args.target_rank),
                          "epoch", str(args.epoch)])

    # Save history
    if not os.path.exists(args.history_path):
        os.makedirs(args.history_path)
    history_loc = os.path.join(args.history_path, args.dataset,'seed_'+str(seed)+'/')
    if not os.path.exists(history_loc):
        os.makedirs(history_loc)
    pd.DataFrame.from_dict(res.history).to_csv(history_loc + '/'+model_str+'.csv', index=False)
    print("Saved history to disk")

    # Save model
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    model_loc = os.path.join(args.model_path, args.dataset,'seed_'+str(seed)+'/')
    if not os.path.exists(model_loc):
        os.makedirs(model_loc)
    model.save(model_loc+'/'+model_str+'.h5')
    print("Saved model to disk")

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    if args.perseed == 1:
        runCNN(args.seed, args)
    else:
        seed_start = int(args.seed_part) * 20
        seed_end = (int(args.seed_part) + 1) * 20
        for i in range(seed_start, seed_end):
            runCNN(i, args)

