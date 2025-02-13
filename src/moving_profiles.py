import jax.numpy as jnp
from scipy.optimize import root_scalar

def get_steps(time_grid, eta, total_time):
    """
    Returns condition arrays based on the scaled time, eta, and T.
    """
    first = (1 - eta) * total_time # 3 * rise time
    second = (1 + 2 * eta) * total_time # 3 * (rise time + move time)
    third = (2 + eta) * total_time # 3 * (rise time + move time + fall time)
    #capture step
    rise = 3 * time_grid < first
    #move step
    move = (first <= 3 * time_grid) & (3 * time_grid < second)
    #release step
    fall = (second <= 3 * time_grid) & (3 * time_grid < third)
    #wait step
    wait = 3 * time_grid >= third
    return rise, move, fall, wait

def magic_poly(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5

def magic_poly_first(s):
    return 30 * s**2 - 60 * s**3 + 30 * s**4

def magic_poly_second(s):
    return 60 * s - 180 * s**2 + 120 * s**3

def generate_linear_profile(mov_amp, time_grid, total_time, ksi_start, ksi_stop, eta):

    move_time = total_time * eta
    rise_time = (total_time - move_time) / 3
    fall_time = rise_time

    amp_profile = jnp.zeros_like(time_grid)
    coord_profile = jnp.zeros_like(time_grid)

    rise, move, fall, wait = get_steps(time_grid, eta, total_time)

    # Generate coordinate profile
    coord_profile = jnp.where(rise, ksi_start, coord_profile) 
    coord_profile = jnp.where(move, ksi_start + (ksi_stop - ksi_start) * (time_grid - rise_time) / move_time, coord_profile)  
    coord_profile = jnp.where(fall, ksi_stop, coord_profile)  
    coord_profile = jnp.where(wait, ksi_stop, coord_profile)

    # Generate amplitude profile
    amp_profile = jnp.where(rise, mov_amp * time_grid / rise_time, amp_profile)  # Rise phase
    amp_profile = jnp.where(move, mov_amp, amp_profile)  # Hold at max value during move phase
    amp_profile = jnp.where(fall, mov_amp * (1 - (time_grid - rise_time - move_time) / fall_time), amp_profile)  # Fall phase
    amp_profile = jnp.where(wait, 0, amp_profile)  # Hold at 0 during wait phase

    return coord_profile, amp_profile

def generate_minjerk_profile(mov_amp, time_grid, total_time, ksi_start, ksi_stop, eta):

    move_time = total_time * eta
    rise_time = (total_time - move_time) / 3
    fall_time = rise_time

    amp_profile = jnp.zeros_like(time_grid)
    coord_profile = jnp.zeros_like(time_grid)

    rise, move, fall, wait = get_steps(time_grid, eta, total_time)

    # Generate coordinate profile
    coord_profile = jnp.where(rise, ksi_start, coord_profile) 

    coord_profile = jnp.where(
        move,  
        ksi_start + (ksi_stop - ksi_start) * magic_poly((time_grid - rise_time)/ move_time), 
        coord_profile
    )  
    coord_profile = jnp.where(fall, ksi_stop, coord_profile)  
    coord_profile = jnp.where(wait, ksi_stop, coord_profile)

    # Generate amplitude profile
    amp_profile = jnp.where(rise, mov_amp * time_grid / rise_time, amp_profile)  # Rise phase
    amp_profile = jnp.where(move, mov_amp, amp_profile)  # Hold at max value during move phase
    amp_profile = jnp.where(fall, mov_amp * (1 - (time_grid - rise_time - move_time) / fall_time), amp_profile)  # Fall phase
    amp_profile = jnp.where(wait, 0, amp_profile)  # Hold at 0 during wait phase

    return coord_profile, amp_profile

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

#args for delta_mt, w_mt, w_st
def generate_profile(profile, amp, time_grid, total_time, x_left, x_right, eta, *args):
    """Generates coordinate and amplitude profiles based on the profile type."""
    if profile == "Linear":
        return generate_linear_profile(amp, time_grid, total_time, x_left, x_right, eta)
    elif profile == "Minjerk":
        return generate_minjerk_profile(amp, time_grid, total_time, x_left, x_right, eta)
    elif profile == "STA":
        return generate_sta_profile(amp, time_grid, total_time, x_left, x_right, eta, *args)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")


