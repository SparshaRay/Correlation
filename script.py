from scipy.optimize import root
import numpy as np

num_sols = 0

for i in range(1001):

    state_vector = np.random.normal(size = 2**3)
    # state_vector = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype = float)
    # state_vector = np.random.rand(2**3)
    state_vector /= np.linalg.norm(state_vector)
    state_vector = abs(state_vector)
    state_vector = np.round(state_vector, 3)
    A = state_vector[0]
    B = state_vector[1]
    C = state_vector[2]
    D = state_vector[3]
    E = state_vector[4]
    F = state_vector[5]
    G = state_vector[6]
    H = state_vector[7]

    def equations(x):

        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]
        e = x[4]
        f = x[5]
        g = x[6]
        h = x[7]

        eq1 = a**2 + b**2 - (A**2 + B**2)
        eq2 = c**2 + d**2 - (C**2 + D**2)
        eq3 = e**2 + f**2 - (E**2 + F**2)
        eq4 = g**2 + h**2 - (G**2 + H**2)
        eq5 = a**2 + c**2 - (A**2 + C**2)
        eq6 = a**2 + e**2 - (A**2 + E**2)
        eq7 = e**2 + g**2 - (E**2 + G**2)

        eq8  = (a*d - a*f - b*c + b*e - c*h + d*g + e*h - f*g) * (a*d + a*f - b*c - b*e + c*h - d*g + e*h - f*g) - (A*D - A*F - B*C + B*E - C*H + D*G + E*H - F*G) * (A*D + A*F - B*C - B*E + C*H - D*G + E*H - F*G)
        eq9  = (a*f - a*g - b*e - b*h + c*e + c*h + d*f - d*g) * (a*f + a*g - b*e + b*h - c*e + c*h - d*f - d*g) - (A*F - A*G - B*E - B*H + C*E + C*H + D*F - D*G) * (A*F + A*G - B*E + B*H - C*E + C*H - D*F - D*G)
        eq10 = (a*d - a*g - b*c - b*h + c*e + d*f + e*h - f*g) * (a*d + a*g - b*c + b*h - c*e - d*f + e*h - f*g) - (A*D - A*G - B*C - B*H + C*E + D*F + E*H - F*G) * (A*D + A*G - B*C + B*H - C*E - D*F + E*H - F*G)

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10]

    sols = []
    sums = []
    sol_count = 0
    # raw = []

    for j in range(100):
        print(f'Iteration {j:02d} of {i}th trial, current number of single solutions : {i - num_sols}/{i}', end='\r')
        sol = root(equations, np.random.rand(8), method='lm')
        if sol.success and sol.x[0] > 0 and sol.x[1] > 0 and sol.x[2] > 0 and sol.x[3] > 0 and sol.x[4] > 0 and sol.x[5] > 0 and sol.x[6] > 0 and sol.x[7] > 0:
            sol_count += 1
            # raw.append(sol.x[0])
            if sum(np.round(sol.x, decimals=3)) not in sums :
                sols.append(sol.x)
                sums.append(sum(np.round(sol.x, decimals=3)))

    if len(sols) > 1 : 
        num_sols += 1
        print(state_vector, *sols, end='\n')
        print(np.linalg.norm(state_vector))
        for sol in sols : print(np.linalg.norm(sol))
        #print(A, raw)
    #if len(sols) > 2 : print(f'\nFatal error, more than 2 solutions found !\n {sols} for {state_vector}\n')

print(f'\nNumber of states with more than 1 solution : {num_sols}/{i}')