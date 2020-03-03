from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
import numpy as np


def _init_conv_weights(option):
    if option == 'high_rank':
        weights = np.random.normal([3,3])
    elif option == 'low_rank':
        weights = np.random.normal([3,2])
        weights = weights.dot(weights.T)
    elif option =='HL':
        h_rank_weights = np.random.normal([3,3])
        l_rank_weights = np.random.normal([3,2])
        l_rank_weights = l_rank_weights.dot(l_rank_weights.T)
        weights = h_rank_weights.dot(l_rank_weights)
    return weights

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

#conv_weights = _init_conv_weights('high_rank')

#add model layers
conv_2d = Conv2D(1, kernel_size=3, activation='relu', input_shape=(28,28,1))
print(conv_2d.get_weights())
model.add(conv_2d)
print(conv_2d.get_weights())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
train_res = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=50)

#predict first 4 images in the test set
eval_res = model.predict(X_test)


print('done')

