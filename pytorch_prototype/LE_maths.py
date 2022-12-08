import numpy as np
import scipy as sp
from scipy import arange, pi, sqrt, zeros
from scipy.special import jv
from scipy.optimize import brentq

from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from scipy.integrate import quadrature

from rascaline import SphericalExpansion
from datetime import datetime

from tqdm import tqdm

def Jn(r, n):
  return (sqrt(pi/(2*r))*jv(n+0.5,r))
def Jn_zeros(n, nt):
  zerosj = zeros((n+1, nt), dtype=np.float64)
  zerosj[0] = arange(1,nt+1)*pi
  points = arange(1,nt+n+1)*pi
  racines = zeros(nt+n, dtype=np.float64)
  for i in range(1,n+1):
    for j in range(nt+n-i):
      foo = brentq(Jn, points[j], points[j+1], (i,))
      racines[j] = foo
    points = racines
    zerosj[i][:nt] = racines[:nt]
  return (zerosj)

def get_spherical_bessel_zeros(l_max, n_max):
    z_ln = Jn_zeros(l_max, n_max)  # Spherical Bessel zeros
    z_nl = z_ln.T
    return z_nl

def get_LE_calculator(l_max, n_max, a, nu, CS, l_nu, l_r):

    date_time = datetime.now()
    date_time = date_time.strftime("%m-%d-%Y-%H-%M-%S-%f")
    spline_file = "splines/splines-" + date_time + ".txt"
    z_nl = get_spherical_bessel_zeros(l_max, n_max)

    def R_nl(n, l, r):
        return j_l(l, z_nl[n, l]*r/a)

    def N_nl(n, l):
        # Normalization factor for LE basis functions
        def function_to_integrate_to_get_normalization_factor(x):
            return j_l(l, x)**2 * x**2
        integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l], miniter=100, maxiter=10000)
        return (1.0/z_nl[n, l]**3 * integral)**(-0.5)

    N_nl_precomputed = np.zeros((n_max, l_max+1))
    for l in range(l_max+1):
        for n in range(n_max):
            N_nl_precomputed[n, l] = N_nl(n, l)

    def get_LE_function(n, l, r):
        R = np.zeros_like(r)
        for i in range(r.shape[0]):
            R[i] = R_nl(n, l, r[i])
        return N_nl_precomputed[n, l]*R*a**(-1.5)

    def sigma(r):  
        # The function that determines how sigma changes as a function of r.
        sigma = CS*np.exp(l_nu*nu+l_r*r)
        return sigma

    from fortran import sbessi
    def exp_i_l(l, x):
        result = np.zeros_like(x)
        for i in range(len(x)):
            result[i] = sbessi(l, x[i])
        return result

    def evaluate_LE_function_mollified_adaptive(n, l, r):
        # Calculates a mollified (but with adaptive sigma) LE radial basis function for a signle value of r.
        c = 1.0/(2.0*sigma(r)**2)
        def function_to_integrate(x):
            return 4.0 * np.pi * x**2 * get_LE_function(n, l, x) * np.exp(-c*(x-r)**2) * exp_i_l(l, 2.0*c*x*r) * (1.0/(np.sqrt(2*np.pi)*sigma(r)))**3
        integral, _ = sp.integrate.quadrature(function_to_integrate, 0.0, a, miniter=100, maxiter=10000)
        return integral

    def get_LE_function_mollified_adaptive(n, l, r):
        # Calculates a mollified (but with adaptive sigma) LE radial basis function for a 1D array of values r.
        R = np.zeros_like(r)
        for i in range(r.shape[0]):
            R[i] = evaluate_LE_function_mollified_adaptive(n, l, r[i])*np.exp(-l_r*r[i])
            # print(r[i], R[i])
        return R

    n_spline_points = 232
    spline_x = np.linspace(0.0, a, n_spline_points)  # x values

    def function_for_splining(n, l, x):
        return get_LE_function_mollified_adaptive(n, l, x)

    spline_f = []
    print("Calculating radial basis values for splining")
    for l in tqdm(range(l_max+1)):
        for n in range(n_max):
            spline_f_single = function_for_splining(n, l, spline_x)
            spline_f.append(spline_f_single)
    spline_f = np.array(spline_f).T
    spline_f = spline_f.reshape(n_spline_points, l_max+1, n_max)  # f(x) values

    def function_for_splining_derivative(n, l, r):
        delta = 1e-6
        all_derivatives_except_first_and_last = (function_for_splining(n, l, r[1:-1]+delta) - function_for_splining(n, l, r[1:-1]-delta)) / (2.0*delta)
        derivative_at_zero = (function_for_splining(n, l, np.array([delta/10.0])) - function_for_splining(n, l, np.array([0.0]))) / (delta/10.0)
        derivative_last = (function_for_splining(n, l, np.array([a])) - function_for_splining(n, l, np.array([a-delta/10.0]))) / (delta/10.0)
        return np.concatenate([derivative_at_zero, all_derivatives_except_first_and_last, derivative_last])

    spline_df = []
    print("Calculating radial basis derivatives for splining")
    for l in tqdm(range(l_max+1)):
        for n in range(n_max):
            spline_df_single = function_for_splining_derivative(n, l, spline_x)
            spline_df.append(spline_df_single)
    spline_df = np.array(spline_df).T
    spline_df = spline_df.reshape(n_spline_points, l_max+1, n_max)  # df/dx values

    with open(spline_file, "w") as file:
        np.savetxt(file, spline_x.flatten(), newline=" ")
        file.write("\n")

    with open(spline_file, "a") as file:
        np.savetxt(file, (1.0/(4.0*np.pi))*spline_f.flatten(), newline=" ")
        file.write("\n")
        np.savetxt(file, (1.0/(4.0*np.pi))*spline_df.flatten(), newline=" ")
        file.write("\n")

    hypers_spherical_expansion = {
            "cutoff": a,
            "max_radial": n_max,
            "max_angular": l_max,
            "center_atom_weight": 0.0,
            "radial_basis": {"Tabulated": {"file": spline_file}},
            "atomic_gaussian_width": 100.0,
            "cutoff_function": {"Step": {}},
        }

    calculator = SphericalExpansion(**hypers_spherical_expansion)
    return calculator
    # spherical_expansion_coefficients = calculator.compute(structures)
