# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성 [X]
    
    for idx in range(x.size):        	
        tmp_val = x[idx]
        print("#_numerical_gradient_no_batch: tmp_val: %d, idx: %d, x.size: %d" % (tmp_val, idx, x.size))
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
#        fxh1 = f(x)		
        fxh1 = f(x[idx])
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
#        fxh2 = f(x)				
        fxh2 = f(x[idx]) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:	# 2차원입니다. [[X] [Y]]
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X): # 2차원을 1차원으로 나눕니다. [X]와 [Y]를 차례대로 읽습니다.
            grad[idx] = _numerical_gradient_no_batch(f, x) #[X]를 f(x)에 입력합니다.
        
        return grad


def function_2(x):
    if x.ndim == 0: # f(x[idx]) 값을 입력할 경우에 0 차원입니다. (숫자)
        return x**2 # 단순 숫자를 제곱하여 리턴합니다.
    if x.ndim == 1: # [X]는 1차원입니다.
        return np.sum(x**2) # [X]의 모든 값들을 제곱하여 더합니다.
		                    # 해당부분은 X**2 + Y**2 을 표시하려고 한 것이나 잘못 사용되고 있습니다.
							# 그러나 기울기 값을 구하면 fxh1 - fxh2 에서 x[idx]+h와 x[idx]-h된 값만 차이가 나므로 상관이 없습니다.
							# f(x)에서 X값을 배열로 전체를 입력하지 않고 x[idx] 하나의 값만 입력하고 retrun x**2로 값의 제곱을 보내도 됩니다.
    else:
        return np.sum(x**2, axis=1) # http://taewan.kim/post/numpy_sum_axis/


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

def test_matrix(x):
    enum = enumerate(x);
    print("#enum: ", enum)	
    for idx, x in enum:
        print("# test_matrix(): idx:", idx)
        print("# x: ", x)	
     
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25) # -2부터 시작해서 2.5 이하까지 0.25 간격으로 전개함. 2.5 이전에 끝남
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1) # (-2,-2),(-2,-1.75),(-2,-1.5),...
    
    X = X.flatten() # [-2,-1.75,..., 2.25, -2,-1.75,..., 2.25, ... 반복]
    Y = Y.flatten() # [-2,-2,...,-2, -1.75,-1.75,...,-1.75, ... 반복]
    A = np.array([X, Y])

    grad = numerical_gradient(function_2, A )	
	
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
