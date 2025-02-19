import jax.numpy as jnp
from scipy.optimize import root_scalar
from profiles.general import get_steps, magic_poly, magic_poly_first, magic_poly_second

def y_func(time_grid, total_time, ksi_start, ksi_stop, eta):
    rise, move, fall, wait = get_steps(time_grid, eta, total_time)
    result = jnp.zeros_like(time_grid)
    result = jnp.where(rise, 0, result)
    result = jnp.where(
        move,
        magic_poly((3 * time_grid - (1 - eta) * total_time) / (3 * eta * total_time)),
        result
    )
    result = jnp.where(fall, 1, result)
    result = jnp.where(wait, 1, result)
    return ksi_start + result * (ksi_stop - ksi_start)

def y_func_prime2(time_grid, total_time, ksi_start, ksi_stop, eta):
    rise, move, fall, wait = get_steps(time_grid, eta, total_time)
    result = jnp.zeros_like(time_grid)
    result = jnp.where(rise, 0, result)
    result = jnp.where(
        move,
        magic_poly_second((3 * time_grid - (1 - eta) * total_time) / (3 * eta * total_time)),
        result
    )
    result = jnp.where(fall, 0, result)
    result = jnp.where(wait, 0, result)
    return result * (ksi_stop - ksi_start) / (eta * total_time) ** 2

def f_func(time_grid, total_time, factor, eta):

    rise, move, fall, wait = get_steps(time_grid, eta, total_time)

    result = jnp.zeros_like(time_grid)
    result = jnp.where(
        rise,
        1 + (factor - 1) * magic_poly(3 * time_grid / ((1 - eta) * total_time)),
        result
    )
    result = jnp.where(move, factor, result)
    result = jnp.where( 
        fall,
        factor - (factor - 1) * magic_poly((3 * time_grid - (1 + 2 * eta) * total_time) / ((1 - eta) * total_time)),
        result
    )
    result = jnp.where(wait, 1, result)
    return result

def f_func_prime2(time_grid, total_time, factor, eta):
    rise, move, fall, wait = get_steps(time_grid, eta, total_time)
    result = jnp.zeros_like(time_grid)
    result = jnp.where(
        rise,
        (factor - 1) * magic_poly_second(3 * time_grid / ((1 - eta) * total_time)),
        result
    )
    result = jnp.where(move, 0, result)
    result = jnp.where( 
        fall,
        -(factor - 1) * magic_poly_second((3 * time_grid - (1 + 2 * eta) * total_time) / ((1 - eta) * total_time)),
        result
    )
    result = jnp.where(wait, 1, result)

    return 9 * result / ((1 - eta) * total_time) ** 2

def k_squared_func(time_grid, total_time, factor, eta):
    f_values = f_func(time_grid, total_time, factor, eta)
    f_prime2_values = f_func_prime2(time_grid, total_time, factor, eta)
    return 1 / f_values**4 - f_prime2_values / (4 * jnp.pi**2 * f_values)

def ksi_0_func(time_grid, total_time, ksi_start, ksi_stop, factor, eta):    
    y = y_func(time_grid, total_time, ksi_start, ksi_stop, eta)
    y_prime2 = y_func_prime2(time_grid, total_time, ksi_start, ksi_stop, eta)
    k_squared = k_squared_func(time_grid, total_time, factor, eta)
    return (4 * jnp.pi ** 2) * y_prime2 / k_squared + y

# for G(x) = exp(-x^2 / 2), return dG(x)/dx
def gauss_prime(ksi):
    return -ksi * jnp.exp(-ksi**2 / 2)

# for G(x) = exp(-x^2 / 2), return d2G(x)/dx2
def gauss_prime2(ksi):
    return (ksi**2 - 1) * jnp.exp(-ksi**2 / 2)

#differentiate static potential derivative at point ksi
def v_st_prime(ksi, ksi_start, ksi_stop):
    return -gauss_prime(ksi - ksi_start) - gauss_prime(ksi - ksi_stop)

#differentiate static potential derivative at point ksi
def v_st_prime2(ksi, ksi_start, ksi_stop):
    return -gauss_prime2(ksi - ksi_start) - gauss_prime2(ksi - ksi_stop)

# we're solving equation:
#  
# ksi - ksi0 - v_st'(ksi) / k^2 = 0
#
# this function is the residual of this equation (e.g. simply its LHS)
def residual(ksi, ksi_start, ksi_stop, ksi_0, k_squared):
    return ksi - ksi_0 - v_st_prime(ksi, ksi_start, ksi_stop) / k_squared

def residual_prime(ksi, ksi_start, ksi_stop, ksi_0, k_squared):
    return 1 - v_st_prime2(ksi, ksi_start, ksi_stop) / k_squared


def generate_sta_profile(
        mov_amp,
        time_grid,
        total_time,
        ksi_start,
        ksi_stop,
        eta,
        delta_mt
    ):

    factor = jnp.sqrt(delta_mt / jnp.sqrt(mov_amp))
    k_squared_values = k_squared_func(time_grid, total_time, factor, eta)
    ksi_0_values = ksi_0_func(time_grid, total_time, ksi_start, ksi_stop, factor, eta)

    amplitudes = []
    ksi_mov = []

    for ksi_0, k_squared in zip(ksi_0_values, k_squared_values):

        solution = root_scalar(
            residual,
            fprime=residual_prime,  
            args=(ksi_start.real, ksi_stop.real, ksi_0.real, k_squared.real),
            x0 = ksi_0
        ).root
        ksi_mov.append(solution)
        amplitude = delta_mt**2 * (k_squared - v_st_prime2(solution, ksi_start, ksi_stop))
        amplitudes.append(amplitude)
        
    return jnp.array(ksi_mov), jnp.array(amplitudes)


