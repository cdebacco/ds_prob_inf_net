import numpy as np
import scipy.optimize
from scipy.stats import rankdata


def sigma_a_metric(A, s, beta, d=1):
    M = np.sum(A)
    assert M > 0  # number of edges
    s = s.reshape((-1, 1))
    P = 1 / (1 + np.exp(-2 * beta * (s - s.T)))
    unnormalized_sigma_a = np.sum(np.abs(A - (A + A.T) * P) ** d)
    return 1 - unnormalized_sigma_a / (2 * M)


def sigma_L_metric(A, s, beta):
    M = np.sum(A)
    assert M > 0
    s = s.reshape((-1, 1))
    unnormalized_sigma_L = np.sum(
        - A * np.logaddexp(0, -2 * beta * (s - s.T)) -
        A.T * np.logaddexp(0, 2 * beta * (s - s.T))
    )
    return unnormalized_sigma_L / M


def beta_a_optimize(A_validation, s, d=1):
    A_validation = np.sum(A_validation, 0)

    def objective(beta, *args):
        s = args[0].reshape((-1, 1))
        assert len(s) == A_validation.shape[1]
        P = 1 / (1 + np.exp(-2 * beta * (s - s.T)))
        assert (np.allclose(P + P.T, 1))
        unnormalized_sigma_a = (np.abs(A_validation - (A_validation + A_validation.T) * P) ** d).sum()
        return unnormalized_sigma_a

    beta_a_opt = scipy.optimize.minimize(objective, args=s, x0=np.array([1.]), method='L-BFGS-B', bounds=[(0, 20)],
                                         options={'ftol': 1.e-12, 'maxiter': 500})['x']
    # print("beta_a_opt=", beta_a_opt)
    return beta_a_opt


def beta_L_optimize(A_validation, s):
    A_validation = np.sum(A_validation, 0)

    def objective(beta, *args):
        s = args[0].reshape((-1, 1))
        assert len(s) == A_validation.shape[1]
        unnormalized_sigma_L = (
                - A_validation * np.logaddexp(0, -2 * beta * (s - s.T)) -
                A_validation.T * np.logaddexp(0, 2 * beta * (s - s.T))
        ).sum()
        return -unnormalized_sigma_L

    beta_L_opt = scipy.optimize.minimize(objective, args=s, x0=np.array([.1]), method='L-BFGS-B', bounds=[(0, 5)],
                                         options={'ftol': 1.e-12, 'maxiter': 500})['x']
    # print("beta_L_opt=", beta_L_opt)
    return beta_L_opt

def _get_ranks(x):
    """ ranks elements in decreasing order, starting from 0 """
    return (rankdata(-x) - 1).astype(int)


def agony(A, s, d=1):
    agony = 0
    M = np.sum(A)
    assert M>0
    ranks = _get_ranks(s)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if ranks[r] - ranks[c] >= 0:
                agony += ((ranks[r] - ranks[c]) ** d) * A[r, c] / M
    return agony
