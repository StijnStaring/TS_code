import casadi as ca
import numpy as np

discrete_values = np.array([1,2,3,4,5])
probability = np.array([0.1,0.2,0.5,0.16,0.04])
opti = ca.Opti()
p = opti.variable()
obj = 0
for j in np.arange(0,len(discrete_values)):
    pi = discrete_values[j]
    ei = probability[j]*((p-pi)/pi)
    Li = opti.variable()
    obj = obj + Li
    opti.subject_to(opti.bounded(-Li,ei,Li))
opti.minimize(obj)
opti.subject_to(p>=0)
opti.set_initial(p,0.3)
opti.solver('ipopt')
sol = opti.solve()
print("This is the value of p: %s." % sol.value(p))

# opti = ca.Opti()
# p = opti.variable(6,1)
# L = opti.variable(2*N+2)
# F = ca.vec(S(p)-D_modified)
#
# opti.minimize(sum1(L))
# opti.subject_to(opti.bounded(-L,F,L))
#
# opti.solver('ipopt')
#
# opti.set_initial(p,p_hat)
#
# sol = opti.solve()
#
# p_fit = sol.value(p)
# print(p_fit)  #   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]
#
# D_fit = S(p_fit)
#
# figure()
# plot(D_fit)
# plot(D_modified)