import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
from six.moves import urllib
import operator
from datetime import datetime

def read_data(bytestream):
    datatype = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=datatype)[0]

def extract_pictures(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zip_file:
        magic_num = read_data(zip_file)
        if magic_num !=2051:
            raise ValueError('在这个MNIST文件中的幻数 %d 无效: %s' %(magic_num, input_file.name))
        num_images = read_data(zip_file)
        rows = read_data(zip_file)
        cols = read_data(zip_file)
        print ('图片数量 %d' %(num_images))
        buffer = zip_file.read(rows * cols * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data

def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic_num = read_data(zipf)
        if magic_num != 2049:
            raise ValueError('在这个MNIST文件中的幻数 %d 无效: %s' % (magic_num, input_file.name))
        num_items = read_data(zipf)
        buffer = zipf.read(num_items)
        labels_data = np.frombuffer(buffer, dtype=np.uint8)
        return labels_data

def knn_train(newInput, dataSet, labels, k, numOfTrain): 
    numSamples = dataSet.shape[0]
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)
    diff = np.tile(newInput, (numOfTrain, 1)) - dataSet[0:numOfTrain]
  
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis = 1)
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


def testMain(numOfTrain, numOfTest, k):
    print ("正在读取训练图片数据...")
    train_x = extract_pictures('data/mnist/train_images', True, True)
    train_y = extract_labels('data/mnist/train_labels')

    print ("正在读取训练测试数据...")
    test_x = extract_pictures('data/mnist/test_images', True, True)
    test_y = extract_labels('data/mnist/test_labels')

    print ("正在训练...")
    a = datetime.now()
    predict = [-1 for i in range(numOfTest)]
    for i in range(numOfTest):
        predict[i] = knn_train(test_x[i], train_x, train_y, k, numOfTrain)
    b = datetime.now()
    print ("训练 使用的时间为%d秒" %((b-a).seconds))

    print ("正在测试...")
    matchCount = 0
    for i in range(numOfTest):
        if predict[i] == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numOfTest
    print ('测试的识别准确率为: %.2f%%' % (accuracy * 100))

if __name__ == '__main__':
    #第一个参数为训练次数，第二个参数为识别次数，第三个参数为KNN中的k值
    testMain(6000, 1000, 3)
