import numpy as np
import pyemma

def adw_potential(x, k=0.0, x0=0.0):
    r"""potential energy of confuguration state x in kT"""
    return 0.5 * (x - 2.0)**4 - 3.0 * (x - 2.0)**2 + x - 2.0 + 0.5 * k * (x - x0)**2

def adw_match_reference_to_binning(xtrajs, clustercenters, kT=1.0):
    r"""reference free energy in kT averaged over discrete states"""
    x = np.linspace(
        np.min([xtraj.min() for xtraj in xtrajs]), np.max([xtraj.max() for xtraj in xtrajs]), 1000)
    d = pyemma.coordinates.assign_to_centers(data=x, centers=clustercenters)[0]
    pi_fine = np.exp(-adw_potential(x) / kT)
    pi_fine /= pi_fine.sum()
    pi_coarse = np.zeros(shape=(clustercenters.shape[0]), dtype=np.float64)
    for i in range(clustercenters.shape[0]):
        idx = np.where(d == i)[0]
        pi_coarse[i] = pi_fine[idx].sum()
    f = -np.log(pi_coarse)
    srt = np.argsort(clustercenters, axis=0)
    return clustercenters[srt, 0], f[srt]

def adw_reference(xmin, xmax, nbins, kT=1.0, k_bias=0.0, x_bias=0.0):
    x = np.linspace(xmin, xmax, nbins)
    delta = 0.5 * (xmax - xmin) / float(nbins)
    y = np.linspace(xmin - delta, xmax + delta, np.max([1000, 20 * nbins]))
    d = pyemma.coordinates.assign_to_centers(data=y, centers=x.reshape((-1, 1)))[0]
    pi_fine = np.exp(-adw_potential(y, k=k_bias, x0=x_bias) / kT)
    pi_fine /= pi_fine.sum()
    pi = np.zeros(shape=(nbins,), dtype=np.float64)
    for i in range(nbins):
        idx = np.where(d == i)[0]
        pi[i] = pi_fine[idx].sum()
    f = -np.log(pi)
    return x, f, pi

def epot_gauss_2d(r, r0, sigma, pf):
    return pf * np.exp(-0.5 * (((r - r0) / sigma)**2).sum())

def epot_harmonic_2d(r, r0, k):
    return 0.5 * (k * (r - r0)**2).sum()

class Hamiltonian(object):
    def __init__(self):
        self.pf = np.array([-8.0, -4.8, -8.0, -4.0], dtype=np.float64)
        self.r0 = np.array([[15.0, 15.0], [9.0, 9.0], [9.0, 21.0], [21.0, 13.0]], dtype=np.float64)
        self.sigma = np.array([10.0, 2.5, 2.5, 2.5], dtype=np.float64)
    def potential_energy(self, r, r0=None, k=None):
        epot = np.sum([epot_gauss_2d(r, self.r0[i], self.sigma[i], self.pf[i]) for i in range(4)])
        if r0 is not None and k is not None:
            epot += epot_harmonic_2d(r, r0, k)
        return epot

def tp_make_centers(n):
    d = np.linspace(5, 25, n, endpoint=False)
    d += 0.5 * (d[1] - d[0])
    x, y = np.meshgrid(d, d)
    return np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)))), d.reshape((-1, 1)).copy()

def tp_reference_2d(nbins_per_axis, RT=1.0, k_bias=None, x_bias=None):
    hamiltonian = Hamiltonian()
    if x_bias is not None:
        r0 = np.array([x_bias, 0])
    else:
        r0 = None
    r, d = tp_make_centers(nbins_per_axis)
    u = np.array([hamiltonian.potential_energy(
        r[i, :], r0=r0, k=k_bias) for i in range(r.shape[0])]) / RT
    pi = np.exp(-u).reshape((nbins_per_axis, nbins_per_axis))
    pi /= pi.sum()
    f = -np.log(pi)
    return r[:, 0].reshape((nbins_per_axis, nbins_per_axis)), \
        r[:, 1].reshape((nbins_per_axis, nbins_per_axis)), f, pi

def tp_match_reference_to_binning(centers):
    hamiltonian = Hamiltonian()
    n = 5 * centers.shape[0]
    centers2d, centers1d = tp_make_centers(n)
    srt = np.argsort(centers, axis=0)
    u2d = np.array(
        [hamiltonian.potential_energy(centers2d[i, :]) for i in range(centers2d.shape[0])])
    pi2d = np.exp(-u2d)
    pi2d /= pi2d.sum()
    pi1d = pi2d.reshape((n, n)).sum(axis=0)
    dtraj = pyemma.coordinates.assign_to_centers(data=centers1d, centers=centers)[0]
    pi_coarse = np.array([np.sum(pi1d[(dtraj == i)]) for i in range(centers.shape[0])])
    return centers[srt, 0], pi_coarse[srt], -np.log(pi_coarse[srt])

def a2_bias_potential(theta_traj, spring_constants, umbrella_centers):
    return spring_constants[np.newaxis, :] * (
        1.0 + np.cos(theta_traj[:, np.newaxis] - umbrella_centers[np.newaxis, :] - np.pi))
