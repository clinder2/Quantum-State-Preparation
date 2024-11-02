import numpy as np
import math
from random import randint

def SA(runs, params, ansatz, simulator):
    #deltaETest = 0
    # average absolute value delta_E: 0.01-0.03
    # acceptance at beginning: 0.99%
    # acceptance at end: 0.00001%
    # temperature: 1.9-0.0017
    theta_vect = params
    def T(t):
        c = 0.02
        a = 0.01
        temperature = c/(a + math.log(t))
        return temperature
    for t in range(1, runs):    
            h = math.pow(math.e, -1 * .1/T(t))
            U = np.random.normal(.5, .5, 1)
            print("h: " + str(h) + ", U: " + str(U) + ", t: " + str(t) + ", T(): " + str(T(t)))
    #print("delta: " + str(deltaETest/runs))
    return params

if __name__ == "__main__":
    randvect = [randint(0,100) for p in range(0,4)]
    norm = np.linalg.norm(randvect)
    randvect = randvect/norm
    print(randvect)
    SA(30, [1,1,1,1], None, None)