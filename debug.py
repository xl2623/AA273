from animations import animate_cartpole
import matplotlib.animation as animation
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
class Particles:
    def __init__(self, μ=np.array([np.inf])):
        if np.sum(μ) == np.inf:
            self.N = 0
            self.x = np.zeros([1,1])
            self.w = np.zeros([1, ])
        else:
            N = np.size(μ, 0)
            self.N = N
            self.x = np.zeros([N, 2*N+1])
            self.w = np.zeros([2*N+1, ])
        
def UT(μ, Σ, λ):
    root = sqrtm(Σ)
    P = Particles(μ)
    N = P.N
    P.x[:, 0] = μ
    P.w[0] = λ/(λ+N)
    for i in range(1, 2*N+1):
        if i <= N:
            P.x[:, i] = μ + np.sqrt(λ+N) * root[:, i-1]
            P.w[i] = 1/(2*(λ+N))
        else:
            P.x[:, i] = μ - np.sqrt(λ+N) * root[:, i-P.N-1]
            P.w[i] = 1/(2*(λ+N))
    return P

def UTdynamics(fd, P, u, dt):
    Pn = Particles(P.x[:, 0])
    Pn.w = P.w
    for i in range(2*Pn.N + 1):
        Pn.x[:, i] = fd(P.x[:, i], u, dt)
    return Pn

def UTdynamicsinv(Pn, Q):
    μ = np.zeros([Pn.N,])
    Σ = np.zeros([Pn.N, Pn.N])
    for i in range(2*Pn.N + 1):
        μ += Pn.w[i] * Pn.x[:, i]
    for j in range(2*Pn.N + 1):
        Σ += Pn.w[j] * np.outer((Pn.x[:, j] - μ), (Pn.x[:, j] - μ))
    Σ += Q
    return μ, Σ

def UTmeasurement(g, P):
    Pn = Particles()
    Pn.w = P.w
    Pn.N = P.N
    Pn.x = np.zeros([np.size(g(P.x[0]), 0), np.size(P.x, 1)])
    for i in range(2*Pn.N + 1):
        Pn.x[:, i] = g(P.x[:, i])
    return Pn

def UTmeasurementinv(Py, Px, R):
    Σxy = np.zeros([np.size(Px.x, 0), np.size(Py.x, 0)])
    Σy = np.zeros([np.size(Py.x, 0), np.size(Py.x, 0)])
    y = np.zeros([np.size(Py.x, 0), ])
    μ, _ = UTdynamicsinv(Px, np.zeros([np.size(Px.x, 0),np.size(Px.x, 0)]))
    for i in range(2*Px.N + 1):
        y += Py.w[i] * Py.x[:, i]
    for j in range(2*Px.N + 1):
        Σy += Py.w[j] * np.outer((Py.x[:, j] - y), (Py.x[:, j] - y).T)
    for k in range(2*Px.N + 1):
        Σxy += Py.w[k] * np.outer((Px.x[:, k] - μ), (Py.x[:, k] - y))
    Σy += R
    return y, Σy, Σxy

# print(np.shape(np.zeros([3,3])))
f = lambda x, u : np.array([u[0] * np.cos(x[2]),u[0] * np.sin(x[2]),u[1]])
g = lambda x: np.array([np.sqrt(x[0]**2+x[1]**2),np.sqrt(x[1]**2)])
fd = lambda μ, u, dt: μ + dt*f(μ, u)



u = np.array([1,1])
dt = 0.1
Q = np.eye(3)
R = np.eye(2)
λ = 2
yt = np.array([[1, 0.5]])

# prediction
P = UT(np.array([1,2,3]), np.eye(3), 2)
fd = lambda μ, u, dt: μ + dt*f(μ, u)
Px = UTdynamics(fd, P, u, dt)
μ, Σ = UTdynamicsinv(Px, Q)

# measurement update
P = UT(μ,Σ,λ)
Py = UTmeasurement(g, P)
y, Σy, Σxy = UTmeasurementinv(Py, Px, R)

μ += np.reshape(Σxy @ inv(Σy) @ ((yt - y).T),[np.size(μ, 0), ])
Σ -= Σxy @ inv(Σy) @ (Σxy.T)

print(μ)
print(Σ)
print(sum(P.w))


def cartpoleReal(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.5     # pendulum mass
    mc = 10.5    # cart mass
    L = 1.2      # pendulum length
    g = 9.81     # gravitational acceleration
    μc = 0.99    # cart ground friction
    μp = 0.99     # cart pole friction

    x, θ, dx, dθ = s

    sinθ, cosθ = np.sin(θ), np.cos(θ)
    ddθ = (-g*sinθ + cosθ*((- u[0] - mp*L*(dθ**2*sinθ) + μc * np.sign(dx)) / (mc + mp))-(μp*dθ)/(mp*L)) / (L*(4/3-(mp*cosθ**2)/(mc+mp)))
    ddx = (mp*L*(dθ**2*sinθ - ddθ*cosθ) + u[0] - μc * np.sign(dx)) / (mc + mp)

    ds = np.array([
        dx,
        dθ,
        ddx,
        ddθ
    ])
    return ds

def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.5     # pendulum mass
    mc = 10.5    # cart mass
    L = 1.2      # pendulum length
    g = 9.81     # gravitational acceleration
    μc = 0.1     # cart ground friction
    μp = 0.1     # cart pole friction

    x, θ, dx, dθ = s
    sinθ, cosθ = np.sin(θ), np.cos(θ)
    # ddθold = (g*sinθ + cosθ*((-u[0]-mp*L*dθ**2*(sinθ + μc*jnp.sign(dx)*cosθ))/(mc+mp)+μc*g*jnp.sign(dx))-(μp*dθ)/(L*mp)) / (L*(4/3-(mp*cosθ)/(mc+mp)*(cosθ - μc * jnp.sign(dx))))
    # Nc = (mc + mp)*g - mp*L*(ddθold * sinθ + dθ**2 * cosθ)
    ddθ = (-g*sinθ + cosθ*((-u[0]-mp*L*dθ**2*(sinθ)/(mc+mp)))) / (L*(4/3-(mp*cosθ**2)/(mc+mp)))
    ddx = (mp*L*(dθ**2*sinθ - ddθ*cosθ) + u[0]) / (mc + mp)
    ds = np.array([
        dx,
        dθ,
        ddx,
        ddθ
    ])
    return ds
n = 4                                       # state dimension
m = 1
T = 10  
t = np.arange(0., T, dt)
N = t.size - 1
s = np.zeros((N + 1, n))
u = np.zeros((N, m))
s0 = np.array([0., 3*np.pi/4, 0., 0.]) 
s[0] = s0
for k in range(N):
    u[k] = 0.
    s[k+1] = odeint(lambda s, t: cartpoleReal(s, u[k]), s[k], t[k:k+2])[1]

print(cartpoleReal(np.array([0., 3*np.pi/4, 0., 0.]) , np.array([0.]) ))

fig, axes = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
plt.subplots_adjust(wspace=0.45)
labels_s = (r'$x(t)$', r'$\theta(t)$', r'$\dot{x}(t)$', r'$\dot{\theta}(t)$')
labels_u = (r'$u(t)$',)
for i in range(n):
    axes[i].plot(t, s[:, i])
    axes[i].set_xlabel(r'$t$')
    axes[i].set_ylabel(labels_s[i])
for i in range(m):
    axes[n + i].plot(t[:-1], u[:, i])
    axes[n + i].set_xlabel(r'$t$')
    axes[n + i].set_ylabel(labels_u[i]) 

fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
writergif = animation.PillowWriter(fps=30)
ani.save('filename.gif',writer=writergif)

plt.show()


    # ddθ = (-g*sinθ + cosθ*((-u[0]-mp*L*dθ**2*(sinθ + μc*np.sign(dx)*cosθ))/(mc+mp)+μc*(-g)*np.sign(dx))-(μp*dθ)/(L*mp)) / (L*(4/3-(mp*cosθ)/(mc+mp)*(cosθ - μc * np.sign(dx))))
    # ddθ = (-g*sinθ + cosθ*((-u[0]-mp*L*dθ**2*(sinθ)/(mc+mp)))) / (L*(4/3-(mp*cosθ**2)/(mc+mp)))
    # Nc = (mc + mp)*(-g) - mp*L*(ddθ * sinθ + dθ**2 * cosθ)
    # # if np.sign(Nc) != np.sign(ddθ)
    # # ddθ = (-g*sinθ + cosθ*((-u[0]-mp*L*dθ**2*(sinθ + μc*np.sign(Nc*dx)*cosθ))/(mc+mp)+μc*(-g)*np.sign(Nc*dx))-(μp*dθ)/(L*mp)) / (L*(4/3-(mp*cosθ)/(mc+mp)*(cosθ - μc * np.sign(Nc * dx))))
    # if np.maximum(np.abs(μc) * Nc, np.abs(mp*L*(dθ**2 * sinθ - ddθ*cosθ) + u[0])) == np.abs(mp*L*(dθ**2 * sinθ - ddθ*cosθ)):
    #     ddx = (u[0]) / (mc + mp)
    # else:
    #     ddx = (mp*L*(dθ**2 * sinθ - ddθ*cosθ) - μc * Nc * np.sign(Nc * dx) + u[0]) / (mc + mp)
    