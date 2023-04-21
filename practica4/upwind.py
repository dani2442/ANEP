import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

a = 1

x_intervalo = (0, 4)
t_intervalo = (0, 1)

Nx = 1000
Nt = 1000

dx = (x_intervalo[1] - x_intervalo[0])/Nx
dt = (t_intervalo[1] - t_intervalo[0])/Nt

x = np.linspace(*x_intervalo, Nx+1)
t = np.linspace(*t_intervalo, Nt+1)

u0_t_inicial = lambda t: 2*t
u0_x_inicial = lambda x: x*(x-2)

X, Y = np.meshgrid(x, t)
u = np.zeros((Nt+1, Nx+1))

u[0] = u0_x_inicial(x)
u[:, 0] = u0_t_inicial(t)

if a > 0:
    print('Upwind')
    alpha = 1 - dt*a/dx
    beta = dt*a/dx

    A = np.diag([alpha]*(Nx+1), 0) + np.diag([beta]*(Nx), -1)
    for i in range(1, len(t)):
        u[i][1:] = np.dot(A, u[i-1])[1:]
    
    
else:
    print('Downwind')
    alpha = (1 - a*(dt/dx))
    beta = -a*(dt/dx)
    A = np.diag([alpha]*Nx, 0) + np.diag([beta]*(Nx-1), 1)
    for i in range(1, len(t)):
        Ai = np.linalg.inv(A)
        u[i][1:] = np.dot(Ai, u[i-1][1:])

plt.contourf(X, Y, u, cmap='hot', levels=100)
plt.colorbar()
plt.show()

    
fig = plt.figure()
camera = Camera(fig)
for i in range(u.shape[0]):
    plt.plot(u[i], c='r')
    camera.snap()
animation = camera.animate(interval=0.01)

plt.show()
    
    