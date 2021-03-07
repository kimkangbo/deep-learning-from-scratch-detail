# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def step_function(x):
    return np.array(x > 0, dtype=np.int) # x>0 이면 True, 아니면 False 임. dtype=np.int로 지정하면 True는 1, False는 0으로 대치됨

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--') # {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}, blue, green, red, cyan, magenta, yellow, black, white
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()
