#!/usr/bin/env python
# encoding: utf-8

"""
@author: izyq
@file: util.py
@time: 2018/4/27 10:05
"""
import numpy as np
# import math

def batch_generator(X, y, batch_size):
    # size=4
    # pickle
    # size=len(X)
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()

    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]

    # print('out:',X_copy)
    i = 0
    while True:
        if i + batch_size <= size:
            # yield
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]

            # print('in',X_copy)
            continue


def batch_generator_predict(X, y, batch_size):
    # size=4
    size = X.shape[0]
    count=0
    i = 0

    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]

    while True:
        if i + batch_size <= size:
            # yield
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
            count+=1
        elif count*batch_size<size:
            yield X_copy[i:],y_copy[i:]
            count+=1
        else:
            count=0
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            # print('in',X_copy)
            continue

# test无需shuffle
def batch_generator_test(X, y,batch_size):
    # size=4
    size = X.shape[0]
    count=0
    i = 0
    while True:
        if i + batch_size <= size:
            # yield
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
            count+=1
        elif count*batch_size<size:
            yield X[i:],y[i:]
            count+=1
        else:
            break

def batch_generator_test_no_answer(X, batch_size):
    # size=4
    size = X.shape[0]
    count=0
    i = 0
    while True:
        if i + batch_size <= size:
            # yield
            yield X[i:i + batch_size]
            i += batch_size
            count+=1
        elif count*batch_size<size:
            yield X[i:]
            count+=1
        else:
            break



# if __name__=="__main__":
#     x=np.array([[1,2,3,0,0],[2,3,4,0,0],[5,6,7,0,0],[7,8,0,0,0],[0,0,0,0,0]])
#     y=np.array([1,2,3,4,5])
#     # batch_data=batch_generator_test(x,y,2)
#     batch_num_in_epochs=int(math.ceil(len(x)/3.0))
#     # for i in range(3):
#     #     for _ in range(batch_num_in_epochs):
#     #         print(batch_data.__next__())
#     #     print('--------------')
#
#     batch_predict=batch_generator_test(x,y,3)
#     for _ in range(batch_num_in_epochs):
#         print(batch_predict.__next__())
#     print('--------------')