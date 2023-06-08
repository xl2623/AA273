"""
Starter code for the problem "Cart-pole swing-up".

Autonomous Systems Lab (ASL), Stanford University
"""

import time

from animations import animate_cartpole
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.integrate import odeint

from numpy.random import multivariate_normal as mvnrnd

def linearize(f, s, u):
    """Linearize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    """
    # WRITE YOUR CODE BELOW ###################################################
    # INSTRUCTIONS: Use JAX to compute `A` and `B` in one line.
    # raise NotImplementedError()
    A, B = jax.jacobian(f, [0,1])(s, u)[0], jax.jacobian(f, [0,1])(s, u)[1]
    ###########################################################################
    return A, B

def ilqr(f, s0, s_goal, N, Q, R, QN, eps=1e-3, max_iters=300000):
    """Compute the iLQR set-point tracking solution.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    Q : numpy.ndarray
        The state cost matrix (2-D).
    R : numpy.ndarray
        The control cost matrix (2-D).
    QN : numpy.ndarray
        The terminal state cost matrix (2-D).
    eps : float, optional
        Termination threshold for iLQR.
    max_iters : int, optional
        Maximum number of iLQR iterations.

    Returns
    -------
    s_bar : numpy.ndarray
        A 2-D array where `s_bar[k]` is the nominal state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u_bar : numpy.ndarray
        A 2-D array where `u_bar[k]` is the nominal control at time step `k`,
        for `k = 0, 1, ..., N-1`
    Y : numpy.ndarray
        A 3-D array where `Y[k]` is the matrix gain term of the iLQR control
        law at time step `k`, for `k = 0, 1, ..., N-1`
    y : numpy.ndarray
        A 2-D array where `y[k]` is the offset term of the iLQR control law
        at time step `k`, for `k = 0, 1, ..., N-1`
    """
    if max_iters <= 1:
        raise ValueError('Argument `max_iters` must be at least 1.')
    n = Q.shape[0]        # state dimension
    m = R.shape[0]        # control dimension

    # Initialize gains `Y` and offsets `y` for the policy
    Y = np.zeros((N, m, n))
    y = np.zeros((N, m))

    # Initialize the nominal trajectory `(s_bar, u_bar`), and the
    # deviations `(ds, du)`
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = f(s_bar[k], u_bar[k])
    ds = np.zeros((N + 1, n))
    du = np.zeros((N, m))

    # iLQR loop
    converged = False
    for counter in range(max_iters):
        # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
        A, B = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
        A, B = np.array(A), np.array(B)

        # PART (c) ############################################################
        # INSTRUCTIONS: Update `Y`, `y`, `ds`, `du`, `s_bar`, and `u_bar`.
        # raise NotImplementedError()
        P = np.zeros((N+1, n,n))
        p = np.zeros((N+1, n))
        eta = np.zeros((N, 1))
        beta = np.zeros((N+1, 1))
        alpha = np.zeros((N+1, 1))

        qT = (QN @ s_bar[N] - QN @ s_goal)
        # alphaT = np.transpose(s_goal - s_bar[N]) @ QN @ (s_goal - s_bar[N])
        alphaT = 1/2 * s_goal.T @ QN @ s_goal + 1/2 * s_bar[N].T @ QN @ s_bar[N] - s_bar[N].T @ QN @ s_goal
        betaT = alphaT

        P[N] = QN
        p[N] = qT
        beta[N] = betaT
        alpha[N] = alphaT

        # backward pass
        # use approximate dynamics
        for k in range(N-1, -1, -1):
            qt = Q @ s_bar[k] - Q @ s_goal
            rt = R @ u_bar[k]

            Hxxt = Q + A[k].T @ P[k+1] @ A[k]
            Huut = R + B[k].T @ P[k+1] @ B[k]
            Hxut = 0 + A[k].T @ P[k+1] @ B[k]
            hxt = qt + A[k].T @ p[k+1]
            hut = rt + B[k].T @ p[k+1]

            Kt = -np.linalg.inv(Huut) @ Hxut.T
            kt = -np.linalg.inv(Huut) @ hut

            P[k] = Hxxt + Hxut @ Kt
            p[k] = hxt + Hxut @ kt

            alphat = 1/2 * s_goal.T @ Q @ s_goal + 1/2 * s_bar[k].T @ Q @ s_bar[k] - s_goal.T @ Q @ s_bar[k] + 1/2 * u_bar[k].T @ R @ u_bar[k]
            alpha[k] = alphat

            eta[k] = alpha[k] + beta[k+1]
            beta[k] = eta[k] + 1/2 * hut.T @ kt

            Y[k] = Kt
            y[k] = kt
        
        # forward
        # use exact dynamics
        for k in range(N):
            # rollout
            du[k] = Y[k] @ ds[k] + y[k]
            ds[k+1] = f(s_bar[k]+ds[k], u_bar[k]+du[k]) - s_bar[k+1]
        
        # update
        s_bar += ds
        u_bar += du
        print(np.max(np.abs(du)))
        print(counter)
        #######################################################################

        if np.max(np.abs(du)) < eps:
            converged = True
            break
    if not converged:
        raise RuntimeError('iLQR did not converge!')
    return s_bar, u_bar, Y, y

def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.     # pendulum mass
    mc = 10.    # cart mass
    L = 1.      # pendulum length
    g = 9.81    # gravitational acceleration

    x, θ, dx, dθ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    h = mc + mp*(sinθ**2)
    ds = jnp.array([
        dx,
        dθ,
        (mp*sinθ*(L*(dθ**2) + g*cosθ) + u[0]) / h,
        -((mc + mp)*g*sinθ + mp*L*(dθ**2)*sinθ*cosθ + u[0]*cosθ) / (h*L)
    ])
    return ds

def cartpoleReal(s, u):
    """Compute the cart-pole state derivative."""
    # mp = 2.2     # pendulum mass
    # mc = 10.0    # cart mass
    # L = 1.0      # pendulum length
    # g = 9.81     # gravitational acceleration
    # μc = 0.1    # cart ground friction
    # μp = 0.0     # cart pole friction


    # # # barely working
    mp = 2.6     # pendulum mass
    mc = 10.7    # cart mass
    L = 1.2      # pendulum length
    g = 9.81     # gravitational acceleration
    μc = 0.3    # cart ground friction
    μp = 0.3     # cart pole friction

    x, θ, dx, dθ = s

    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    ddθ = (-g*sinθ + cosθ*((- u[0] - mp*L*(dθ**2*sinθ) + μc * jnp.sign(dx)) / (mc + mp))-(μp*dθ)/(mp*L)) / (L*(4/3-(mp*cosθ**2)/(mc+mp)))
    ddx = (mp*L*(dθ**2*sinθ - ddθ*cosθ) + u[0] - μc * jnp.sign(dx)) / (mc + mp)

    ds = jnp.array([
        dx,
        dθ,
        ddx,
        ddθ
    ])
    return ds

def At(f,
      μ: np.ndarray,
      u: np.ndarray,
      dt: float):
    A, _ = linearize(lambda s, u: s +dt*f(s, u), μ, u)
    return A

def Ct(g,
       μ: np.ndarray):
    C = jax.jacobian(g, 0)(μ)
    return C

def predict(# change each iter
        μ: np.ndarray,  # current mean
        u: float,       # control
        dt: float,      # dt
        Σ: np.ndarray,  # current cov
        # may be static
        f,              # dynamics
        A,              # dynamics jacobian
        Q: np.ndarray): # process noise
    # predict
    At = A(f, μ, u, dt)
    μ = μ + dt*f(μ, u)
    Σ = At @ Σ @ At.T + Q
    return μ, Σ

def EKF(# change each iter
        μ: np.ndarray,  # current mean
        u: float,       # control
        dt: float,      # dt
        Σ: np.ndarray,  # current cov
        yt: np.ndarray,  # current measurement
        # may be static
        f,    # dynamics
        g,    # measurement module
        A,    # dynamics jacobian
        C,    # measruement jacobian
        Q: np.ndarray,  # process noise
        R: np.ndarray): # measurment noise
    # predict
    At = A(f, μ, u, dt)
    μ = μ + dt*f(μ, u)
    Σ = At @ Σ @ At.T + Q

    # update
    Ct = C(g, μ)
    print(Ct)
    K = Σ @ Ct.T @ inv(Ct @ Σ @ Ct.T + R)
    μ += K @ (yt - g(μ))
    Σ -= K @ Ct @ Σ
    return μ, Σ

def UKF(λ: float,        # tuning param
        # change each iter
        μ: np.ndarray,  # current mean
        u: float,       # control
        dt: float,      # dt
        yt: np.ndarray,  # current measurement
        # may be static
        f,    # dynamics
        g,    # measurement module
        Σ: np.ndarray,  # current cov
        Q: np.ndarray,  # process noise
        R: np.ndarray): # measurment noise
    
    class SimgaPoints:
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
        P = SimgaPoints(μ)
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
        Pn = SimgaPoints(P.x[:, 0])
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
        Pn = SimgaPoints()
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
    
    # prediction
    P = UT(μ, Σ, λ)
    fd = lambda μ, u, dt: μ + dt*f(μ, u)
    Px = UTdynamics(fd, P, u, dt)
    μ, Σ = UTdynamicsinv(Px, Q)

    # measurement update
    P = UT(μ,Σ,λ)
    Py = UTmeasurement(g, P)
    y, Σy, Σxy = UTmeasurementinv(Py, Px, R)

    μ += np.reshape(Σxy @ inv(Σy) @ ((yt - y).T),[np.size(μ, 0), ])
    Σ -= Σxy @ inv(Σy) @ (Σxy.T)

    return μ, Σ

def iEKF(# change each iter
        μ: np.ndarray,  # current mean
        u: float,       # control
        dt: float,      # dt
        Σ: np.ndarray,  # current cov
        yt: np.ndarray,  # current measurement
        # may be static
        tol: float,      # converge tolorance
        maxstep: int,    # maximum steps
        f,    # dynamics
        g,    # measurement module
        A,    # dynamics jacobian
        C,    # measruement jacobian
        Q: np.ndarray,  # process noise
        R: np.ndarray): # measurment noise
    # predict
    At = A(f, μ, u, dt)
    μ = μ + dt*f(μ, u)
    Σ = At @ Σ @ At.T + Q

    # update
    converged = False
    μprev = μ
    μcurr = np.zeros(np.shape(μprev))
    count = 0
    while not converged and count < maxstep:
        Ct = C(g, μ)
        K = Σ @ Ct.T @ inv(Ct @ Σ @ Ct.T + R)
        μcurr = μ + K @ (yt - g(μ)) + K @ Ct @ (μprev - μ)
        if (abs(μcurr - μprev) < tol).all():
            converged = True
        count += 1
        μprev = μcurr

    print("here")
    μ = μcurr
    Σ = Σ - K @ Ct @ Σ
    return μ, Σ

# def PF( # change each iter
#         μ: np.ndarray,  # current mean
#         u: float,       # control
#         dt: float,      # dt
#         yt: np.ndarray,  # current measurement
#         # may be static
#         f,    # dynamics
#         g,    # measurement module
#         Σ: np.ndarray,  # current cov
#         Q: np.ndarray,  # process noise
#         R: np.ndarray): # measurment noise:

def model_diff(cartpole, cartpoleReal, n, delay, Σ = 2*np.eye(4)):
    def kl_mvn(m_to, S_to, m_fr, S_fr):
        """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
        d = m_fr - m_to
        
        c, lower = scipy.linalg.cho_factor(S_fr)
        def solve(B):
            return scipy.linalg.cho_solve((c, lower), B)
        
        def logdet(S):
            return np.linalg.slogdet(S)[1]

        term1 = np.trace(solve(S_to))
        term2 = logdet(S_fr) - logdet(S_to)
        term3 = d.T @ solve(d)
        return (term1 + term2 + term3 - len(d))/2.

    # create state centered at 0^T (randomly drawn)
    toStates = []
    cartpoleDynamics = []
    cartpoleRealDynamics = []
    
    for i in range(n):
        toStates.append(mvnrnd(np.array([0,0,0,0]), Σ))
    # ax.scatter([state[0] for state in toStates], [state[1] for state in toStates], color='r')
    
    for state in toStates:
        # cartpoleDynamics.append(state+dt*cartpole(state, np.array([100,100])))
        cartpoleDynamics.append(odeint(lambda s, t: cartpole(s, np.array([100,100])), state, np.array([0.,0.1]))[1])

    for state in toStates:
        # cartpoleRealDynamics.append(state+dt*cartpoleReal(state, np.array([100,100])))
        cartpoleRealDynamics.append(odeint(lambda s, t: cartpoleReal(s, np.array([100,100])), state, np.array([0.,0.1-delay]))[1])

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.scatter([state[2] for state in cartpoleDynamics], [state[3] for state in cartpoleDynamics], color='b')
    # ax.scatter([state[2] for state in cartpoleRealDynamics], [state[3] for state in cartpoleRealDynamics], color='r')
    # plt.show()

    μfr = sum(cartpoleDynamics)/n
    Σfr = sum([np.outer((state - μfr),(state - μfr)) for state in cartpoleDynamics])/n
    μto = sum(cartpoleRealDynamics)/n
    Σto = sum([np.outer((state - μto),(state - μto)) for state in cartpoleRealDynamics])/n



    return kl_mvn(μto, Σto, μfr, Σfr)

def compute_cost(s, u):
    s_goal = np.array([0., np.pi, 0., 0.])
    Q = np.diag(np.array([10., 10., 2., 2.]))
    R = 1e-2*np.eye(1)                          # control cost matrix
    QN = 1e4*np.eye(4)
    QN[0,0] = 1e2
    QN[2,2] = 1e2                          # terminal state cost matrix
    cost = 0
    for i in range(s.shape[0]-1):
        cost += (s[i]-s_goal) @ Q @ np.transpose((s[i]-s_goal)) + u[i] @ R @ np.transpose(u[i])
    cost += (s[-1]-s_goal) @ QN @ np.transpose((s[-1]-s_goal))
    return 0.5*cost

np.random.seed(seed=1)
# Define constants
n = 4                                       # state dimension
m = 1                                       # control dimension
Q = np.diag(np.array([10., 10., 2., 2.]))   # state cost matrix
R = 1e-2*np.eye(m)                          # control cost matrix
QN = 1e2*np.eye(n)                          # terminal state cost matrix
s0 = np.array([0., 0., 0., 0.])             # initial state
s_goal = np.array([0., np.pi, 0., 0.])      # goal state
T = 10.                                     # simulation time
dt = 0.1                                    # sampling time
Rm = np.eye(2)*1e-3                         # Measurement noise
Qm = np.eye(n)                              # Process noise
Σ0 = np.eye(n)*0.1                          # initial belief
μ0 = np.array([0., 0., 0., 0.])             # initial mean
fullstatefb = True
filteron = False                             # use or not use filter

filter = "ekf"
delay = 0.03                                   # number of dt delayed
designctrl = False                           # then perform iLQR
compute_model_diff = True                  # compnute model diff using KL div and odeint

graphics = True






# Initialize continuous-time and discretized dynamics
f = jax.jit(cartpole)
freal = jax.jit(cartpoleReal)
# fd = jax.jit(lambda s, u, dt=dt: discretize(cartpole, dt, s, u))
fd = jax.jit(lambda s, u, dt=dt: s + dt*f(s, u))

# Initialize measurement dynamics
g = jax.jit(lambda x: jnp.array([x[1],x[3]]))

# Compute the iLQR solution with the discretized dynamics
t = np.arange(0., T, dt)
N = t.size - 1
if designctrl:
    print('Computing iLQR solution ... ', end='', flush=True)
    start = time.time()
    s_bar, u_bar, Y, y = ilqr(fd, s0, s_goal, N, Q, R, QN)
    print('done! ({:.2f} s)'.format(time.time() - start), flush=True)
    Y_reshaped = Y.reshape(Y.shape[0], -1)
    np.savetxt('s_bar.txt', s_bar)
    np.savetxt('u_bar.txt', u_bar)
    np.savetxt('bigyvalue.txt', Y_reshaped)
    np.savetxt('yvalue.txt', y)
else:
    Ypholder = np.zeros((N, m, n))
    s_bar = np.zeros((N + 1, n))
    s_bar = np.loadtxt('s_bar.txt', dtype=float)
    u_bar = np.loadtxt('u_bar.txt', dtype=float)
    loaded_arr = np.loadtxt('bigyvalue.txt', dtype=float)
    Y = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // Ypholder.shape[2], Ypholder.shape[2])
    y = np.loadtxt('yvalue.txt', dtype=float)
print('####################################################')
if compute_model_diff:
    print('computing model diff using KL div...')
    print('Dynamics difference measured by KL divergence is: '+ str(model_diff(cartpole, cartpoleReal, 100, delay)))
else:
    print('skipping model diff computation')

print('####################################################')
# Simulate on the true continuous-time system
if filteron:
    print('Current Setup: Full state feedback:', fullstatefb, ', delay:', str(delay), "sec, using", filter)
else:
    print('Current Setup: Full state feedback:', fullstatefb, ', delay:', str(delay), "sec, no filter")

print('Simulating ... ', end='', flush=True)
start = time.time()
s = np.zeros((N + 1, n))
# sdelay = np.zeros((N+1, n))
u = np.zeros((N, m))
# s0 = np.array([0., 0., 0., 0.]) 
s[0] = s0
μ = np.zeros((N + 1, n))
Σ = np.zeros((N + 1, n, n))
μ[0] = μ0
Σ[0] = Σ0
delayedmeasurement = np.array([0., 0., 0., 0.])
for k in range(N):
    if filteron:
        u[k] = u_bar[k] + y[k] +Y[k] @ (μ[k] - s_bar[k])
        # ctrl delay
        # actual state
        s[k+1] = odeint(lambda s, t: freal(s, u[k]), s[k], t[k:k+2])[1]
        # measurement, est, ctrl delay
        delayedmeasurement = odeint(lambda s, t: freal(s, u[k]), s[k+1], np.array([t[k+1], t[k+1]-delay]))[1]
        yt = g(delayedmeasurement) + mvnrnd(np.zeros(np.size(g(delayedmeasurement), 0)), Rm)

        if filter == "ukf":
            μ[k+1], Σ[k+1] = UKF(   λ=2,        # tuning param
                                    # change each iter
                                    μ=μ[k],  # current mean
                                    u=u[k],       # control
                                    Σ=Σ[k],  # current cov
                                    dt=dt,      # dt
                                    yt=yt,  # current measurement
                                    # may be static
                                    f=f,    # dynamics
                                    g=g,    # measurement module
                                    Q=Qm,  # process noise
                                    R=Rm) # measurment noise
        elif filter == "ekf":
            μ[k+1], Σ[k+1] = EKF(# change each iter
                                    μ=μ[k],  # current mean
                                    u=u[k],       # control
                                    Σ=Σ[k],  # current cov
                                    dt=dt,      # dt
                                    yt=yt,  # current measurement
                                    # may be static
                                    f=f,    # dynamics
                                    g=g,    # measurement module
                                    A=At,    # dynamics jacobian
                                    C=Ct,    # measruement jacobian
                                    Q=Qm,  # process noise
                                    R=Rm) # measurment noise
        elif filter == "iekf":
            μ[k+1], Σ[k+1] = iEKF(# change each iter
                                    μ=μ[k],  # current mean
                                    u=u[k],       # control
                                    Σ=Σ[k],  # current cov
                                    dt=dt,      # dt
                                    yt=yt,  # current measurement
                                    # may be static
                                    tol=1e-5,      # converge tolorance
                                    maxstep=1e2,    # maximum steps
                                    f=f,    # dynamics
                                    g=g,    # measurement module
                                    A=At,    # dynamics jacobian
                                    C=Ct,    # measruement jacobian
                                    Q=Qm,  # process noise
                                    R=Rm) # measurment noise

    elif fullstatefb:
        u[k] = u_bar[k] + y[k] +Y[k] @ (delayedmeasurement + mvnrnd(np.array([0,0,0,0]), 1e-4*np.eye(4)) - s_bar[k])

        # u[k] = 0.
        s[k+1] = odeint(lambda s, t: freal(s, u[k]), s[k], t[k:k+2])[1]
        # measurement, est, ctrl delay
        delayedmeasurement = odeint(lambda s, t: freal(s, u[k]), s[k+1], np.array([t[k+1], t[k+1]-delay]))[1]
    
    else: # no full state feedback, just predict no measuremnet
        u[k] = u_bar[k] + y[k] +Y[k] @ (μ[k] - s_bar[k])
        s[k+1] = odeint(lambda s, t: freal(s, u[k]), s[k], t[k:k+2])[1]
        μ[k+1], Σ[k+1] = predict(# change each iter
                                μ=μ[k],      # current mean
                                u=u[k],      # control
                                dt=dt,       # dt
                                Σ=Σ[k],      # current cov
                                # may be static
                                f=f,         # dynamics
                                A=At,        # dynamics jacobian
                                Q=Qm)        # process noise



print('done! ({:.2f} s)'.format(time.time() - start), flush=True)
print('cost is:', compute_cost(s, u))

if graphics:
    # Plot and animate
    if filteron or not fullstatefb:
        fig, axes = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
        plt.subplots_adjust(wspace=0.45)
        labels_s_y = (r'$x(t)$', r'$\theta(t)$', r'$\dot{x}(t)$', r'$\dot{\theta}(t)$')
        labels_s = (r'$x(t)$ True', r'$x(t)$ Est', r'$\theta(t)$ True', r'$\theta(t)$ Est', r'$\dot{x}(t)$ True', r'$\dot{\theta}(t)$ True', r'$\dot{\theta}(t)$ Est')
        labels_u = (r'$u(t)$',)
        counter = 0
        for i in range(n):
            axes[i].plot(t, s[:, i], label=labels_s[counter])
            counter += 1
            axes[i].plot(t, μ[:, i], label=labels_s[counter])
            ci = 1.96 * np.sqrt(Σ[:,i,i])
            axes[i].fill_between(t, (μ[:, i]-ci), (μ[:, i]+ci), color='b', alpha=.1, label="95% CI")
            axes[i].set_xlabel(r'$t$')
            axes[i].set_ylabel(labels_s_y[i])
            axes[i].legend()
            axes[i].set_xlim([0, 10])
            if i == 0:
                axes[i].set_ylim([-3, 5])
            elif i == 1:
                axes[i].set_ylim([-2, 4.14])
            elif i == 2:
                axes[i].set_ylim([-6, 3])
            elif i == 3:
                axes[i].set_ylim([-2, 5])
        for i in range(m):
            axes[n + i].plot(t[:-1], u[:, i])
            axes[n + i].set_xlabel(r'$t$')
            axes[n + i].set_ylabel(labels_u[i])
    else:
        fig, axes = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
        plt.subplots_adjust(wspace=0.45)
        labels_s = (r'$x(t)$', r'$\theta(t)$', r'$\dot{x}(t)$', r'$\dot{\theta}(t)$')
        labels_u = (r'$u(t)$',)
        for i in range(n):
            axes[i].plot(t, s[:, i])
            axes[i].set_xlabel(r'$t$')
            axes[i].set_ylabel(labels_s[i])
            axes[i].set_xlim([0, 10])
            if i == 0:
                axes[i].set_ylim([-3, 5])
            elif i == 1:
                axes[i].set_ylim([-2, 4.14])
            elif i == 2:
                axes[i].set_ylim([-6, 3])
            elif i == 3:
                axes[i].set_ylim([-2, 5])
        for i in range(m):
            axes[n + i].plot(t[:-1], u[:, i])
            axes[n + i].set_xlabel(r'$t$')
            axes[n + i].set_ylabel(labels_u[i]) 
    plt.savefig('cartpole_swingup_cl.png', bbox_inches='tight')

    plt.savefig('cartpole_swingup_cl.png', bbox_inches='tight')

    fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
    writergif = animation.PillowWriter(fps=50)
    ani.save('filename.gif',writer=writergif)

    plt.show()