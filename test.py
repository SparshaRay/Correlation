# from mpmath import findroot
# import numpy as np
# from numpy import conjugate as cj

# state_vector = np.random.rand(2**3) + 1j*np.random.rand(8) - 0.5 - 0.5j
# state_vector /= np.linalg.norm(state_vector)
# print(state_vector)
# A = state_vector[0]
# B = state_vector[1]
# C = state_vector[2]
# D = state_vector[3]
# E = state_vector[4]
# F = state_vector[5]
# G = state_vector[6]
# H = state_vector[7]

# def eq1(a,b,c,d,e,f,g,h) : return a*cj(a) + b*cj(b) - (A*cj(A) + B*cj(B))
# def eq2(a,b,c,d,e,f,g,h) : return c*cj(c) + d*cj(d) - (C*cj(C) + D*cj(D)) 
# def eq3(a,b,c,d,e,f,g,h) : return e*cj(e) + f*cj(f) - (E*cj(E) + F*cj(F))
# def eq4(a,b,c,d,e,f,g,h) : return g*cj(g) + h*cj(h) - (G*cj(G) + H*cj(H))
# def eq5(a,b,c,d,e,f,g,h) : return a*cj(a) + c*cj(c) - (A*cj(A) + C*cj(C))
# def eq6(a,b,c,d,e,f,g,h) : return a*cj(a) + e*cj(e) - (A*cj(A) + E*cj(E))
# def eq7(a,b,c,d,e,f,g,h) : return e*cj(e) + g*cj(g) - (E*cj(E) + G*cj(G))
# def eq8(a,b,c,d,e,f,g,h) : return a*cj(b) + e*cj(f) - (A*cj(b) + E*cj(F))

# root = findroot([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8], list(np.random.rand(2**3)), tol=1e-3, method='muller')
# print(root)

import numpy as np
from numpy import conjugate as cj
from scipy.optimize import root

state_vector = np.random.rand(2**3) + 1j*np.random.rand(8) - 0.5 - 0.5j
state_vector /= np.linalg.norm(state_vector)

A = state_vector[0]
B = state_vector[1]
C = state_vector[2]
D = state_vector[3]
E = state_vector[4]
F = state_vector[5]
G = state_vector[6]
H = state_vector[7]

def equations(x):

    a = x[0]  + 1j*x[1]
    b = x[2]  + 1j*x[3]
    c = x[4]  + 1j*x[5]
    d = x[6]  + 1j*x[7]
    e = x[8]  + 1j*x[9]
    f = x[10] + 1j*x[11]
    g = x[12] + 1j*x[13]
    h = x[14] + 1j*x[15]

    eq1 = a*cj(a) + b*cj(b) - (A*cj(A) + B*cj(B))
    eq2 = c*cj(c) + d*cj(d) - (C*cj(C) + D*cj(D))
    eq3 = e*cj(e) + f*cj(f) - (E*cj(E) + F*cj(F))
    eq4 = g*cj(g) + h*cj(h) - (G*cj(G) + H*cj(H))
    eq5 = a*cj(a) + c*cj(c) - (A*cj(A) + C*cj(C))
    eq6 = a*cj(a) + e*cj(e) - (A*cj(A) + E*cj(E))
    eq7 = e*cj(e) + g*cj(g) - (E*cj(E) + G*cj(G))

    eq8  = a*cj(b) + e*cj(f) - (A*cj(b) + E*cj(F))
    eq9  = a*cj(e) + c*cj(g) - (A*cj(E) + C*cj(G))
    eq10 = a*cj(c) + b*cj(d) - (A*cj(C) + B*cj(D))

    return [eq1.real, eq1.imag, eq2.real, eq2.imag, eq3.real, eq3.imag, eq4.real, eq4.imag, eq5.real, eq5.imag, eq6.real, eq6.imag, eq7.real, eq7.imag, eq8.real, eq8.imag, eq9.real, eq9.imag, eq10.real, eq10.imag]


sol = root(equations, np.random.rand(16), method='lm')
print(f'{state_vector}\n{sol.x}\n{sol.success}')
