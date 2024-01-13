from sympy import *
import numpy as np
from sympy import symbols
parent = np.zeros((8, 8))

init_printing() 

for i in range(1, 256) :
    coeff = symbols(f'a{i}')
    arr = np.array([int(c) for c in f'{i:08b}'])
    mat = np.outer(arr, arr)

    parent = parent + coeff * mat

print(parent)

