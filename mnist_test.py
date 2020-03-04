from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import asarray
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
from keras.initializers import glorot_uniform
import numpy as np
import os
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep CNN.")
    parser.add_argument('--option', nargs='?', default='h_rank',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--target_rank', type=int, default=2,
                        help='Target rank for Conv low rank weight matrix.')
    parser.add_argument('--dataset', nargs='?', default='mnist',
                        help='Choose a Dataset')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epoch')
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
def _init_conv_weights(option, num_kernels, kernel_row, kernel_col, num_channels, target_rank):
    if option == 'h_rank':
        weights = np.random.normal(size=(kernel_row, kernel_col, num_channels, num_kernels))
    elif option == 'l_rank':
        sub_weights = np.random.normal(size=(kernel_row,target_rank))
        weights = sub_weights.dot(sub_weights.T)
        weights = np.reshape(weights, (kernel_row, kernel_col, num_channels, num_kernels))
    elif option =='HL':
        h_rank_weights = np.random.normal(size=(kernel_row, kernel_col))
        l_rank_weights = np.random.normal(size=(kernel_row,target_rank))
        l_rank_weights = l_rank_weights.dot(l_rank_weights.T)
        weights = h_rank_weights.dot(l_rank_weights)
        weights = np.reshape(weights, (kernel_row, kernel_col, num_channels, num_kernels))
    elif option =='l_rank_identity':
        sub_weights = get_identity_matrix(kernel_row, target_rank)
        weights = sub_weights.dot(sub_weights.T)
        weights = np.reshape(weights, (kernel_row, kernel_col, num_channels, num_kernels))
    elif option == 'HL_identity':
        h_rank_weights = np.random.normal(size=(kernel_row, kernel_col))
        l_rank_weights = get_identity_matrix(kernel_row, target_rank)
        l_rank_weights = l_rank_weights.dot(l_rank_weights.T)
        weights = h_rank_weights.dot(l_rank_weights)
        weights = np.reshape(weights, (kernel_row, kernel_col, num_channels, num_kernels))



    return weights

def runCNN(args):
    #download mnist data and split into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #plot the first image in the dataset
    # plt.imshow(X_train[0])
    # plt.show()

    print(X_train[0].shape)
    #reshape data to fit model
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_train[0]

    #create model
    model = Sequential()

    #add model layers
    num_channels = 1
    num_kernels = 1
    kernel_row = 3
    kernel_col = 3
    target_rank = 2
    conv_2d = Conv2D(num_kernels, kernel_size=3, activation='relu', input_shape=(28, 28, num_channels))
    model.add(conv_2d)
    weights = _init_conv_weights(args.option, num_kernels, kernel_row, kernel_col, num_channels, target_rank)
    conv_2d.set_weights([weights, asarray([0.0])])
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    res = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=args.epoch)
    model_str = args.dataset +'_' +str(args.option)

    # Save history
    if not os.path.isdir("./history/"):
        os.mkdir("./history/")
    with open('./history/history_'+model_str+'.json', 'w') as file:
        json.dump(str(res.history), file)
    print("Saved history to disk")

    # Save model
    if not os.path.isdir("./pretrain/"):
        os.mkdir("./pretrain/")
    model.save("./pretrain/"+model_str+'.h5')
    print("Saved model to disk")

    # Save figures
    if not os.path.isdir("./result_figures/"):
        os.mkdir("./result_figures/")
    # summarize history for accuracy
    plt.plot(res.history['accuracy'])
    plt.plot(res.history['val_accuracy'])
    plt.title('model accuracy ('+model_str+')')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./result_figures/accuracy_"+model_str+".png")
    plt.clf()
    # summarize history for loss
    plt.plot(res.history['loss'])
    plt.plot(res.history['val_loss'])
    plt.title('model loss ('+model_str+')')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./result_figures/loss_" + model_str + ".png")


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    runCNN(args)

