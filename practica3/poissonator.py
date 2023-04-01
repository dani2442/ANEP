import numpy as np
import matplotlib.pyplot as plt



M=5
N=7

x_lim = (0,0.5)
y_lim = (0,0.5)

dx = (x_lim[1]-x_lim[0])/(M-1)
dy = (y_lim[1]-y_lim[0])/(N-1)

x = np.linspace(*x_lim,M)
y = np.linspace(*y_lim,N)
    
u_0y = lambda y: 0*y
u_x0 = lambda x: 0*x
u_x1 = lambda x: 200*x
u_1y = lambda y: 200*y


B = np.diag([-2]*(M-2), 0) + np.diag([1]*(M-3), 1) + np.diag([1]*(M-3), -1)

A_x = np.kron(np.eye(N-2, dtype=int), B)
A_y = np.diag([1]*((M-2)*(N-2)-(M-2)), M-2) + np.diag([1]*((M-2)*(N-2)-(M-2)), -(M-2)) + np.diag([-2]*((N-2)*(M-2)),0)

A = A_x/dx**2 + A_y/dy**2


b_x = np.zeros((N-2,M-2))
b_y = np.zeros((N-2,M-2))

b_y[0,:] = u_x1(x[1:-1])
b_y[-1,:] = u_x0(x[1:-1])

b_x[:,0] = u_0y(y[1:-1])[::-1]
b_x[:,-1] = u_1y(y[1:-1])[::-1]

b = -b_x/dx**2 - b_y/dy**2

u_sol = np.linalg.solve(A,b.reshape(-1,1))
u_sol = u_sol.reshape(N-2,M-2)

real_sol = lambda x,y: 400*x*y

X, Y = np.meshgrid(x[1:-1],y[1:-1])
S = real_sol(X, Y)[::-1]


plt.matshow(u_sol)
plt.matshow(S)

print(np.max(np.abs(u_sol-S)))
#plt.matshow(A)
plt.colorbar()
plt.show()
    
    