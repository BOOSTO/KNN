import numpy as np
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

def knn(data, point, k, sigma, p):
    data_trim = np.array(data[:,:-1])
    point = np.array(point)

    # compute distance
    distances = []
    if p != 2:
        distances = np.power(np.sum(np.power(np.absolute(data_trim - point), p), axis = 1), 1 / p)
    else:
        distances = np.linalg.norm(data_trim - point, axis=1)

    neighbors = []
    if k < len(distances):
        # get indices of k nearest neighbors
        checklist = distances
        for i in range(k):
            min = np.where(checklist == np.amin(checklist))
            neighbors.append(min[0][0])
            checklist = np.delete(checklist, neighbors[i], 0)
    else:
        k = len(distances)
        neighbors = range(k)

    # get weights of neighbors
    weight_n = []
    for i in range(k):
        weight_n.append(weight(distances[neighbors[i]], sigma))

    # get classifications of neighbors
    class_n = []
    for i in range(k):
        class_n.append(data[neighbors[i]][-1])
        
    # classify point
    v1 = 0
    v0 = 0
    for i in range(k):
        if class_n[i] == 1:
            v1 = v1 + weight_n[i]
        else:
            v0 = v0 + weight_n[i]

    # return classificaiton
    if (v1 > v0): return 1
    return 0

def fourfold(k, sigma, p):
    d_train = np.genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    np.random.shuffle(d_train)

    # construct 4 slices
    slices = []
    slice_sz = int(len(d_train) / 4)
    for i in range(4):
        s = d_train[i * slice_sz:(i+1) * slice_sz - 1]
        slices.append(s)

    # perform 4-fold cross validation
    results = []
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
            if knn(t[:,1:], v[j,1:-1], k, sigma, p) == v[j,-1]:
                correct = correct + 1
        
        accuracy = correct / total
        print("accuracy of fold[" + str(i) + "]: " + str(accuracy))
        results.append(accuracy)
    
    return results
              
def testknn(k, sigma, p):
    d_train = np.genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    correct = 0

    for i in range(len(d_train)):
        d_point = d_train[i]
        d_data = np.delete(d_train, i, 0)
        if knn(d_data[:,1:], d_point[1:-1], k, sigma, p) == d_point[-1]:
            correct = correct + 1
    return correct / len(d_train)

def eval():
    f = open("results.csv", "w")
    f.write("k,training_acc,fold_mean,fold_var\n")
    for k in [1, 3, 5, 7, 9, 99, 999, 8000]:
        print("\n-----------------------------------------------")
        #run training accuracy test:
        print("training accuracy k = " + str(k) + ", sigma = 1.5, p = 2:")
        acc_results = testknn(k, 1.5, 2)
        print("accuracy: " + str(acc_results))

        # run 4-fold cross validation
        print("\n4-fold with k = " + str(k) + ", sigma = 1.5, p = 2:")
        fold_results = fourfold(k, 1.5, 2)
        mean = np.mean(fold_results)
        var = np.var(fold_results)
        print("\nmean of cross validation: " + str(mean))
        print("variance of cross validation: " + str(var))
        f.write(str(k) + "," + str(acc_results) + "," + str(mean) + "," + str(var) + "\n")
    f.close()

def hyperparamsearch():
    f = open("paramsearch.csv", "w")
    f.write("k,sigma,p,fold_mean,fold_var\n")

    maxmean = 0
    k_optimal = 0
    s_optimal = 0
    p_optimal = 0
    for k in [3, 4, 5]:
        for i in range(50):
            sigma = 0.75 + i * 0.01
            for j in [-1, 0, 2]:
                p = math.pow(2, j)

                # run 4-fold cross validation
                print("\n4-fold with k = " + str(k) + ", sigma = " + str(sigma) + ", p = " + str(p) + ":")
                fold_results = fourfold(k, sigma, p)
                mean = np.mean(fold_results)
                var = np.var(fold_results)

                #update optimals if better
                if mean > maxmean:
                    maxmean = mean
                    k_optimal = k
                    s_optimal = sigma
                    p_optimal = p

                print("\nmean of cross validation: " + str(mean))
                f.write(str(k) + "," + str(sigma) + "," + str(p) + "," + str(mean) + "," + str(var) + "\n")
    print("\noptimal k: " + str(k_optimal) + " optimal sigma: " + str(s_optimal))
    print("mean accuracy: " + str(maxmean))
    f.close()

def kaggle(k, sigma, p):
    d_train = np.genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    d_test = np.genfromtxt('data/test_pub.csv', delimiter=',', skip_header=1)
    f = open("kaggle.csv", "w")
    f.write("id,income\n")
    for point in d_test:
        idx = int(point[0])
        income = knn(d_train[:,1:], point[1:], k, sigma, p)
        f.write(str(idx) + "," + str(income) + "\n")
    f.close()

#kaggle(4, 1.27, 2)
hyperparamsearch()
#eval()