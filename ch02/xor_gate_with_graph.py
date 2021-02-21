# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from and_gate import AND
from or_gate import OR
from nand_gate import NAND


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def makeData():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    tpData = []
	
    for i in range(len(x)):	
        for j in range(len(y)):
            tp = (x[i], y[j])
            tpData.append(tp)
		
    print ("tpData: %s\n", tpData)
	
    return tpData	
	
if __name__ == '__main__':
    tpData = makeData()
    results = []
    for xs in tpData:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
        results.append(y)		
	
    mark = ""	
    for i in range(len(tpData)):	
        x = tpData[i][0]	
        y = tpData[i][1]	
        result = results[i]		
        if result==1:
            mark = "ro"		
        else:
            mark = "bx"		
        plt.plot(x,y, mark)
		
    plt.legend()		
    plt.show()