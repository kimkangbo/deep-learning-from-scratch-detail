# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0
start = time.time()
for i in range(0, len(x), batch_size):	# range(start, stop, step), batch_size 만큼 i의 값을 더함
    x_batch = x[i:i+batch_size]			# batch_size 만큼 (100개씩) x값의 list, 즉 x_batch 리스트를 생성함
    y_batch = predict(network, x_batch) # x_batch를 입력해서 y_batch 리스트로 리턴함
    p = np.argmax(y_batch, axis=1)		# https://gomguard.tistory.com/145 , 100Kx10의 행렬값을 axis=1(10열) 기준으로 가장 일치울이 큰것으로 해서 100x1 행렬로 바꿈
    accuracy_cnt += np.sum(p == t[i:i+batch_size])	# 100x1의 행렬을 실제 값과 비교함 100개의 값이 일치하는 true(1) (false이면 0))을 Sum함
print("time :", time.time() - start)	# 하나씩 실행할때 소요시간: 0.363, 100개씩 배치 실행: 0.026, 14배나 빠름
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
