import jax.numpy as jnp

def get_steps(time_grid, eta, total_time):

    first = (1 - eta) * total_time  # 3 * rise time
    second = (1 + 2 * eta) * total_time  # 3 * (rise time + move time)
    third = (2 + eta) * total_time  # 3 * (rise time + move time + fall time)
    
    rise = 3 * time_grid < first
    move = (first <= 3 * time_grid) & (3 * time_grid < second)
    fall = (second <= 3 * time_grid) & (3 * time_grid < third)
    wait = 3 * time_grid >= third
    
    return rise, move, fall, wait

def time_bounds(total_time, eta):

    move_time = total_time * eta
    rise_time = (total_time - move_time) / 3
    fall_time = rise_time
    return move_time, rise_time, fall_time


def profile_template(mov_amp, time_grid, total_time, ksi_start, ksi_stop, eta):

    move_time, rise_time, fall_time = time_bounds(total_time, eta)
    distance = ksi_stop - ksi_start

    amp_profile = jnp.zeros_like(time_grid)
    coord_profile = jnp.zeros_like(time_grid)

    rise, move, fall, wait = get_steps(time_grid, eta, total_time)

    # Generate coordinate profile
    coord_profile = jnp.where(rise, 0, coord_profile) 
    coord_profile = jnp.where(move, 0, coord_profile)  
    coord_profile = jnp.where(fall, distance, coord_profile)  
    coord_profile = jnp.where(wait, distance, coord_profile)
    coord_profile += ksi_start

    # Generate amplitude profile
    amp_profile = jnp.where(rise, mov_amp * time_grid / rise_time, amp_profile)  
    amp_profile = jnp.where(move, mov_amp, amp_profile) 
    amp_profile = jnp.where(fall, mov_amp * (1 - (time_grid - rise_time - move_time) / fall_time), amp_profile)  
    amp_profile = jnp.where(wait, 0, amp_profile) 

    return coord_profile, amp_profile

def magic_poly(s):
    return s**3 * (10 + s * (-15 + s * 6))

def magic_derivative(s):
    return s**2 * (30 + s * (-60 + s * 30))

def magic_derivative2(s):
    return s * (60 + s * (-180 + s * 120))
