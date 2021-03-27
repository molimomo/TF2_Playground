import pandas as pd
import numpy as np
import math
def get_identity_matrix(rows, cols):
    res = np.zeros((rows, cols))
    row_bound = int(rows/2)
    col_bound = int(cols/2)
    for i in range(rows):
        for j in range(cols):
            if (i < row_bound and j < col_bound) or (i >= row_bound and j >= col_bound) :
                res[i, j] = 1
    return res


def get_identity_matrix_2(rows, cols):
    res = np.zeros((rows, cols))
    row_bound = int(rows/2)
    col_bound = int(cols/2)
    for i in range(rows):
        for j in range(cols):
            if (i < row_bound and j < col_bound) or (i >= row_bound and j >= col_bound) :
                res[i, j] = 1
    np_res = np.array(res)
    np_res[np_res==1] = 2
    print('dsds')

    return res


def get_identity_matrix_3(rows, cols):
    res = np.zeros((rows, cols))
    g = math.floor(rows/cols)
    mode = 0 # regular
    for i in range(rows):
        if mode == 0:
            non_zero_start = i * g
            non_zero_end = (i+1) * g

            if (non_zero_end >= cols):
                mode = 1
                g_beta = cols % g
                if g_beta == 0:
                    g_beta = g
                for j in range(non_zero_start, cols):
                    res[i, j] = 1 / math.sqrt(g_beta)
            else:
                for j in range(non_zero_start, non_zero_end):
                    res[i, j] = 1 / math.sqrt(g)
        else:
            g_beta = cols % g
            if g_beta == 0:
                g_beta = g
            for j in range(non_zero_start, cols):
                res[i, j] = 1 / math.sqrt(g_beta)
        #print(f"start: {non_zero_start}, end: {non_zero_end}")
    final = np.matmul(res, res.transpose())
    return res

def get_identity_matrix_4(rows, cols):
    res = np.zeros((rows, cols))
    g = math.floor(rows/cols)
    
    for i in range(cols):
        non_zero_start = i * g
        non_zero_end = (i + 1) * g
        if non_zero_end >= rows:
            non_zero_end = rows
        for j in range(non_zero_start, non_zero_end):
            res[j, i] = 1
        if i == cols - 1:  # the last column
            for j in range(non_zero_start + 1, rows):
                res[j, i] = 1
    final = np.matmul(res, res.transpose())
    print('dsds')


def get_identity_matrix_5(rows, cols):
    res = np.zeros((rows, cols))
    g = math.floor(rows / cols)

    for i in range(cols):
        non_zero_start = i * g
        non_zero_end = (i + 1) * g
        if non_zero_end >= rows:
            non_zero_end = rows
        for j in range(non_zero_start, non_zero_end):
            res[j, i] = 1/math.sqrt(g)
        if i == cols - 1:  # the last column
            for j in range(non_zero_start, rows):
                res[j, i] = 1/math.sqrt(rows - non_zero_start)
    final = np.matmul(res, res.transpose())
    print('dsds')

#dentity_mat = get_identity_matrix_5(11, 5)
#identity_mat = get_identity_matrix_5(2273, 512)
identity_mat = get_identity_matrix_5(5, 3)
# a = [[1, 0, 0],
#      [1, 0, 0],
#      [0,0.5, 0.5],
#      [0,0.5, 0.5],
#      [0,0.5, 0.5]]
# a = [[1, 0, 0],
#      [1, 0, 0],
#      [0, 1, 0.5],
#      [0, 1, 0.5],
#      [0, 0.5, 0.5]]
#
# np_a = np.array(a)
#
# b = np_a.dot(np_a.T)
#
# print('dsd')
