from numpy import *

def knn(data, point, k):
    print("training data:")
    print(data)

def runknn():
    d_train = genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    d_test = genfromtxt('data/test_pub.csv', delimiter=',', skip_header=1)

    knn(d_train, d_test[0], 4)

runknn()