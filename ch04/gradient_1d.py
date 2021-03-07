# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x           # https://dmcyong.tistory.com/entry/%EC%A0%91%EC%84%A0%EC%9D%98-%EB%B0%A9%EC%A0%95%EC%8B%9D            	
    return lambda t: d*t + y # https://www.w3schools.com/python/python_lambda.asp return tf(t) = (d*t + y) = d*t + (f(x)-d*x) = d*(t-x) + f(x)
	                         # (y-f(a))/(x-a) = f'(a)
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
