# example of average pooling
# from numpy import asarray
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import AveragePooling2D
import numpy as np

def get_identity_matrix(rows, cols):
    res = np.zeros((rows, cols))
    row_bound = int(rows/2)
    col_bound = int(cols/2)
    for i in range(rows):
        for j in range(cols):
            if (i < row_bound and j < col_bound) or (i >= row_bound and j >= col_bound) :
                res[i, j] = 1
    return res


res = get_identity_matrix(5, 3)
fin_res = np.matmul(res, res.T)

print('dads')

# # define input data
# data = [[0, 0, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 1, 1, 0, 0, 0]]
# data = asarray(data)
# data = data.reshape(1, 8, 8, 1)
# # create model
# model = Sequential()
# model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# model.add(AveragePooling2D())
# # summarize model
# model.summary()
# # define a vertical line detector
# detector = [[[[0]],[[1]],[[0]]],
#             [[[0]],[[1]],[[0]]],
#             [[[0]],[[1]],[[0]]]]
#
#
#
# weights = [asarray(detector), asarray([0.0])]
# # store the weights in the model
# model.set_weights(weights)
# # apply filter to input data
# yhat = model.predict(data)
# # enumerate rows
# for r in range(yhat.shape[1]):
# 	# print each column in the row
# 	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])