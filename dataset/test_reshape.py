import numpy as np
    
def test():
    x = np.arange(0, 20, 1)
    print("x\n", x)	
    x1 = x.reshape(-1, 4)
    print("x\n", x)		
    x2 = x.reshape(-1, 2, 2)
    print("x.reshape(-1, 2, 2)\n", x2)
    x3 = x1.reshape(-1, 2, 2)
    print("x1.reshape(-1, 2, 2)\n", x3)	
    x4 = x1.reshape(-1, 1, 2, 2)
    print("x1.reshape(-1, 1, 2, 2)\n", x4)
    x5 = x1.reshape(-1, 2, 2, 1)
    print("x1.reshape(-1, 2, 2, 1)\n", x5)	
	
if __name__ == '__main__':
    test()