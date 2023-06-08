import scipy
import numpy as np
from numpy.random import multivariate_normal as mvnrnd
import matplotlib.pyplot as plt

def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.     # pendulum mass
    mc = 10.    # cart mass
    L = 1.      # pendulum length
    g = 9.81    # gravitational acceleration

    x, θ, dx, dθ = s
    sinθ, cosθ = np.sin(θ), np.cos(θ)
    h = mc + mp*(sinθ**2)
    ds = np.array([
        dx,
        dθ,
        (mp*sinθ*(L*(dθ**2) + g*cosθ) + u[0]) / h,
        -((mc + mp)*g*sinθ + mp*L*(dθ**2)*sinθ*cosθ + u[0]*cosθ) / (h*L)
    ])
    return ds

def cartpoleReal(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.5     # pendulum mass
    mc = 10.7    # cart mass
    L = 1.2      # pendulum length
    g = 9.81     # gravitational acceleration
    μc = 0.5    # cart ground friction
    μp = 0.5     # cart pole friction

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

def model_diff(cartpole, cartpoleReal, n, Σ = 2*np.eye(4)):
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

    dt = 0.1
    # create state centered at 0^T (randomly drawn)
    toStates = []
    cartpoleDynamics = []
    cartpoleRealDynamics = []
    
    for i in range(n):
        toStates.append(mvnrnd(np.array([0,0,0,0]), Σ))
    # ax.scatter([state[0] for state in toStates], [state[1] for state in toStates], color='r')
    
    for state in toStates:
        cartpoleDynamics.append(state+dt*cartpole(state, np.array([100,100])))
    

    for state in toStates:
        cartpoleRealDynamics.append(state+dt*cartpoleReal(state, np.array([100,100])))

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


print(model_diff(cartpole, cartpoleReal, 10000))