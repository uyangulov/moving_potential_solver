import jax.numpy as jnp
from profiles.misc import get_steps

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