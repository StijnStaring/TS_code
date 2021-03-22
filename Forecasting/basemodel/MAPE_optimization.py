import casadi as ca
import numpy as np

discrete_values = np.array([1,2,3,4,5])
probability = np.array([0.1,0.2,0.5,0.16,0.04])
opti = ca.Opti()
p = opti.variable()
obj = 0
for j in np.arange(0,len(discrete_values)):
    pi = discrete_values[j]
    obj = obj + probability[j]*((p-pi)/pi)**2
opti.minimize(obj)
opti.subject_to(p>=0)
opti.set_initial(p,0.3)
opti.solver('ipopt')
sol = opti.solve()
print("This is the value of p: %s." % sol.value(p))

