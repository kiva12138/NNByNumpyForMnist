import os
import struct
import numpy as np
from datetime import datetime

def load_data(path,kind="train"):
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte"%kind)
    images_path = os.path.join(path,"%s-images.idx3-ubyte"%kind)
    with open(labels_path,'rb') as labpath:
        struct.unpack(">II",labpath.read(8))
        labels = np.fromfile(labpath,dtype=np.uint8)
    with open(images_path,'rb') as imgpath:
        struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8)
    return labels,images

def one_hot_code(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def initialize_hidden_layers(num_hidden_units):
    arg1 = np.random.randn(num_hidden_units,784)
    arg2 = np.zeros((num_hidden_units,1))
    arg3 = np.random.randn(10,num_hidden_units)
    arg4 = np.zeros((10,1))
    parameters={"arg1":arg1,"arg2":arg2,"arg3":arg3,"arg4":arg4}
    return parameters

def alive_hidden(num):
    return 1/(1+np.exp(-num)) 

def alive_output(num):
    return np.exp(num)/np.sum(np.exp(num),axis=0,keepdims=True)

def forward_propagation(input_x,output_y,parameters):
    m = input_x.shape[1]
    arg1 = parameters["arg1"]
    arg2 = parameters["arg2"]
    arg3 = parameters["arg3"]
    arg4 = parameters["arg4"]
    a1 = alive_hidden(np.dot(arg1,input_x)+arg2)
    a2 = alive_output(np.dot(arg3,a1)+arg4)
    value_cost = -1/m*np.sum(output_y*np.log(a2))
    return a1,a2,value_cost

def backward_propagation(input_x,output_y,parameters,learning_rate,iterations):
    m = input_x.shape[1]
    arg1 = parameters["arg1"]
    arg2 = parameters["arg2"]
    arg3 = parameters["arg3"]
    arg4 = parameters["arg4"]
    for i in range(iterations):
        a1,a2,cost = forward_propagation(input_x,output_y,parameters)
        dz2 = a2-output_y
        darg3 = 1/m*np.dot(dz2,a1.T)
        darg4 = 1/m*np.sum(dz2,axis=1,keepdims=True)
        dz1 = 1/m*np.dot(arg3.T,dz2)*a1*(1-a1)
        darg1 = 1/m*np.dot(dz1,input_x.T)
        darg2 = 1/m*np.sum(dz1,axis=1,keepdims=True)
        arg1 = arg1-learning_rate*darg1
        arg2 = arg2-learning_rate*darg2
        arg3 = arg3-learning_rate*darg3
        arg4 = arg4-learning_rate*darg4
        parameters={"arg1":arg1,"arg2":arg2,"arg3":arg3,"arg4":arg4}
    return parameters

def predict_action(input_x,output_y,parameters):
    m = output_y.shape[1]
    _,y_hat,_ =forward_propagation(input_x,output_y,parameters)
    y_predict = np.eye(10)[np.array(y_hat.argmax(0))]
    return y_predict.T

def get_accuracy(y_predict,output_y, test_num):
    m = output_y.shape[1]
    n = output_y.shape[0]
    num_correct = 0
    for i in range (0, n-1):
        for j in range (0, m-1):
            if output_y[i][j]==y_predict[i][j]:
                num_correct = num_correct+1
                if(test_num == i*(n+1)+j):
                    return num_correct/((i+1)*(j+1))
    return num_correct/(m*n)

def build_model(input_x,output_y,hidden_units,learning_rate,iterations):
    parameters = initialize_hidden_layers(hidden_units)
    parameters = backward_propagation(input_x,output_y,parameters,learning_rate,iterations)
    return parameters

def testMain(hidden_units_num, learning_rate, iteration_num, test_num):
    print("开始读入训练图片数据...")
    orig_labels,orig_images = np.array(load_data("data/",kind="train"))
    images = orig_images.reshape(60000,784)/255
    images = images.T
    mid_labels = one_hot_code(orig_labels,10)
    labels =mid_labels.T
    print("图片数量："+str(images.shape[1]))

    print("开始读入测试图片数据...")
    orig_test_labels,orig_test_images = np.array(load_data("data/",kind="t10k"))
    test_images = orig_test_images.reshape(10000,784)/255
    test_images = test_images.T
    mid_test_labels = one_hot_code(orig_test_labels,10)
    test_labels = mid_test_labels.T
    print("图片数量："+str(test_images.shape[1]))

    np.random.seed(1)
    a = datetime.now()
    print("开始训练...")
    parameters = build_model(images,labels,hidden_units_num,learning_rate,iteration_num)
    b = datetime.now()
    print ("训练使用的时间为%d秒" %((b-a).seconds))
    
    print("开始测试...")
    test_y_prediction = predict_action(test_images,test_labels,parameters)
    test_accuracy = get_accuracy(test_y_prediction,test_labels,test_num)
    print("测试集的准确率为：%.2f%%" %(test_accuracy*100))

if __name__ == '__main__':
    #第一参数为隐藏层单元数，第二个参数为学习率，第三个参数为训练次数，第四个参数为识别次数
    testMain(100, 0.6, 150, 7500)
