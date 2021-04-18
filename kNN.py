from numpy import *
import math
import random

def inlist(n, list):
    for i in list:
        if (i == n):
            return True
    return False

def weight(d, sigma):
    exponent = -1.0 * (math.pow(d, 2) / sigma)
    return pow(math.e, exponent)

def knn(data, point, k, sigma):
    data_trim = array(data[:,:-1])
    point = array(point)

    # compute distance
    distances = linalg.norm(data_trim - point, axis=1)
    
    # get indices of k nearest neighbors
    neighbors = []
    for i in range(k):
        min = math.inf
        for j in range(len(distances)):
            if min == math.inf:
                if not inlist(j, neighbors):
                    min = j
            elif distances[j] < distances[min] and not inlist(j, neighbors):
                min = j
        neighbors.append(min)

    # get weights of neighbors
    weight_n = []
    for i in range(k):
        weight_n.append(weight(distances[neighbors[i]], sigma))

    # get classifications of neighbors
    class_n = []
    for i in range(k):
        class_n.append(data[neighbors[i]][-1])
        
    # classify point
    class_p = 0
    for i in range(k):
        class_p = class_p + class_n[i] * weight_n[i]
    class_p = class_p / k

    # print("debug:")
    # for i in range(k):
    #     print("\nneighbor [" + str(i) + "]:")
    #     print(class_n[i])
    #     print(weight_n[i])
    # print("\nclassification:")
    # print(class_p)

    # return classificaiton
    if (class_p >= 0.5): return 1
    return 0

def fourfold(k, sigma):
    d_train = genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    random.shuffle(d_train)

    # construct 4 slices
    slices = []
    slice_sz = int(len(d_train) / 4)
    for i in range(4):
        s = d_train[i * slice_sz:(i+1) * slice_sz - 1]
        slices.append(s)

    # perform 4-fold cross validation
    for i in range(4):
        # create train and validation set
        t = []
        v = []
        for j in range(4):
            if j == i:
                if not t:
                    t = slices[j]
                else:
                    t = concatenate((t, slices[j]), axis=0)
            else:
                v = slices[j]

        correct = 0
        total = len(v)
        #run knn for each validation datum
        for j in range(len(v)):
            if knn(t[:,1:], v[j,1:-1], k, sigma) == v[j,-1]:
                correct = correct + 1
        
        accuracy = correct / total
        print("accuracy of fold[" + str(i) + "]: " + str(accuracy))
              
def runknn():
    d_train = genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    d_test = genfromtxt('data/test_pub.csv', delimiter=',', skip_header=1)

    # call kNN without ids or classifier
    knn(d_train[:,1:], d_test[0,1:], 8)

for k in [1, 3, 5, 7, 9]:
    for sigma in [0.1, 1, 10, 100]:
        print("\n4-fold with k = " + str(k) + " and sigma = " + str(sigma) + ":")
        fourfold(k, sigma)
#runknn()