from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow import keras
from tensorflow.keras import layers
import math
#tf.compat.v1.disable_eager_execution()
def parse_args():
    parser = argparse.ArgumentParser(description="Run Feed forward NN.")
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
                        help='run with single seed:1, run with multiple seeds:0')
    parser.add_argument('--identity_option', type=int, default=1,
                        help='0: unscalaring, 1: scalar to 1')
    return parser.parse_args()

# l_rank: W = V * V^T
class l_rank_2(layers.Layer):
    def __init__(self, input_dim=32, rank=10):
        super(l_rank_2, self).__init__()
        self.v = self.add_weight(name='l_rank_v',
                                 shape=(input_dim, rank),
                                 initializer="random_normal",
                                 trainable=True)
        self.w = self.add_weight(name='l_rank_w',
                                 shape=(1,input_dim),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):
        print(f"input shape: {inputs.shape}")
        vvt = tf.matmul(self.v, tf.transpose(self.v))
        vvtw = tf.matmul(vvt, tf.transpose(self.w))
        y = tf.matmul(inputs, vvtw)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'v': self.v
        })
        return config

# l_rank: W = V * V^T
class l_rank(layers.Layer):
    def __init__(self, input_dim=32, rank=10):
        super(l_rank, self).__init__()
        self.v = self.add_weight(name='l_rank_v',
                                 shape=(input_dim, rank),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):
        vvt = tf.matmul(self.v, tf.transpose(self.v))
        y = tf.matmul(inputs, vvt)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'v': self.v
        })
        return config

class h_rank(layers.Layer):
    def __init__(self, input_dim=32):
        super(h_rank, self).__init__()
        self.w = self.add_weight(name='h_rank_w',
                                 shape=(input_dim, input_dim),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):
        y = tf.matmul(inputs, self.w)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w': self.w
        })
        return config

class HL(layers.Layer):
    def __init__(self, input_dim=32, rank=10):
        super(HL, self).__init__()
        self.w = self.add_weight(name='h_rank_w',
                                 shape=(input_dim, input_dim),
                                 initializer="random_normal",
                                 trainable=True)

        self.v = self.add_weight(name='l_rank_v',
                                 shape=(input_dim, rank),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):
        vvt = tf.matmul(self.v, tf.transpose(self.v))
        wvvt = tf.matmul(vvt, self.w)
        y = tf.matmul(inputs, wvvt)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w': self.w,
            'v': self.v
        })
        return config

class l_rank_identity(layers.Layer):
    def __init__(self, input_dim=32, rank=10):
        super(l_rank_identity, self).__init__()
        self.v = self.add_weight(name='l_rank_identity_v',
                                 shape=(input_dim, rank),
                                 initializer="random_normal",
                                 trainable=True)
        self.v.assign(get_identity_matrix(input_dim, rank))


    def call(self, inputs):
        print(inputs.shape)
        print(self.v.shape)
        vvt = tf.matmul(self.v, tf.transpose(self.v))
        y = tf.matmul(inputs, vvt)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'v': tf.make_ndarray(self.v)
        })
        return config

class HL_identity(layers.Layer):
    def __init__(self, input_dim=32, rank=10):
        super(HL_identity, self).__init__()
        self.w = self.add_weight(name='h_rank_w',
                                 shape=(input_dim, input_dim),
                                 initializer="random_normal",
                                 trainable=True)

        self.v = self.add_weight(name='l_rank_identity_v',
                                 shape=(input_dim, rank),
                                 initializer="random_normal",
                                 trainable=True)
        self.v.assign(get_identity_matrix(input_dim, rank))

    def call(self, inputs):
        vvt = tf.matmul(self.v, tf.transpose(self.v))
        wvvt = tf.matmul(vvt, self.w)
        y = tf.matmul(inputs, wvvt)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w': self.w,
            'v': self.v
        })
        return config

def get_identity_matrix(rows, cols):
    res = np.zeros((rows, cols))
    g = math.floor(rows / cols)

    for i in range(cols):
        non_zero_start = i * g
        non_zero_end = (i + 1) * g
        if non_zero_end >= rows:
            non_zero_end = rows
        for j in range(non_zero_start, non_zero_end):
            res[j, i] = 1 / math.sqrt(g)
        if i == cols - 1:  # the last column
            for j in range(non_zero_start, rows):
                res[j, i] = 1 / math.sqrt(rows - non_zero_start)
    return res


def load_datasets(dataset):
    if dataset == 'mnist':
        return mnist.load_data()
    elif dataset == 'fashion_mnist':
        return fashion_mnist.load_data()
    elif dataset == 'cifar10':
        return cifar10.load_data()
    elif dataset == 'cifar100':
        return cifar100.load_data()


def load_model(config):
    model = None
    option = config['option']
    num_features = config['num_features']
    num_classes = config['num_classes']
    target_rank = config['target_rank']

    if option == 'l_rank':
        model = Sequential([
            keras.Input(shape=(num_features,)),
            l_rank(input_dim=num_features, rank=target_rank),
            Dense(1, name='linear_layer'),
            Dense(num_classes, activation='softmax'),
        ])
    elif option == 'h_rank':
        model = Sequential([
            keras.Input(shape=(num_features,)),
            h_rank(input_dim=num_features),
            Dense(1, name='linear_layer'),
            Dense(num_classes, activation='softmax'),
        ])
    elif option == 'HL':
        model = Sequential([
            keras.Input(shape=(num_features,)),
            HL(input_dim=num_features, rank=target_rank),
            Dense(1, name='linear_layer'),
            Dense(num_classes, activation='softmax'),
        ])
    elif option == 'l_rank_identity':
        model = Sequential([
            keras.Input(shape=(num_features,)),
            l_rank_identity(input_dim=num_features, rank=target_rank),
            Dense(1, name='linear_layer'),
            Dense(num_classes, activation='softmax'),
        ])
    elif option == 'HL_identity':
        model = Sequential([
            keras.Input(shape=(num_features,)),
            HL_identity(input_dim=num_features, rank=target_rank),
            Dense(1, name='linear_layer'),
            Dense(num_classes, activation='softmax'),
        ])
    elif option == 'baseline':
        model = Sequential([
            keras.Input(shape=(num_features,)),
            Dense(1, name='linear_layer'),
            Dense(num_classes, activation='softmax'),
        ])
    return model


def run_feed_forward_nn(seed, args):
    # configuration
    config = dict()

    # set random state
    tf.random.set_seed(seed)

    # load dataset
    (X_train, y_train), (X_test, y_test) = load_datasets(args.dataset)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    # flattern dataset
    num_features = X_train.shape[1] * X_train.shape[2]
    num_classes = len(np.unique(np.concatenate((np.unique(y_train), np.unique(y_test)))))
    X_train = X_train.reshape(-1, num_features)
    X_test = X_test.reshape(-1, num_features)

    # one-hot encoding for label
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    config['option'] = args.option
    config['target_rank'] = args.target_rank
    config['num_features'] = num_features
    config['num_classes'] = num_classes

    # build model
    model = load_model(config)

    # compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    res = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=args.epoch)

    ww = model.summary()
    print(ww)

    model_str = "_".join([args.option,
                          "rank", str(args.target_rank),
                          "epoch", str(args.epoch)])
    # Save history_old
    if not os.path.exists(args.history_path):
        os.makedirs(args.history_path)
    history_loc = os.path.join(args.history_path, args.dataset, 'seed_' + str(seed) + '/')
    if not os.path.exists(history_loc):
        os.makedirs(history_loc)
    pd.DataFrame.from_dict(res.history).to_csv(history_loc + '/' + model_str + '.csv', index=False)
    print("Saved history to disk")



    # # Save model
    # if not os.path.isdir(args.model_path):
    #     os.makedirs(args.model_path)
    # model_loc = os.path.join(args.model_path, args.dataset, 'seed_' + str(seed) + '/')
    # if not os.path.exists(model_loc):
    #     os.makedirs(model_loc)
    # model.save(model_loc + '/' + model_str + '.h5')
    # print("Saved model to disk")


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    if args.perseed == 1:
        run_feed_forward_nn(args.seed, args)
    else:
        seed_start = int(args.seed_part) * 10
        seed_end = (int(args.seed_part) + 1) * 10
        for i in range(seed_start, seed_end):
            run_feed_forward_nn(i, args)
#
# x = tf.ones((1, 4))
# model = Sequential([
#             keras.Input(shape=(5,), name='input'),
#             Dense(4, name='layer1'),
#         ])
# print(model.summary())

