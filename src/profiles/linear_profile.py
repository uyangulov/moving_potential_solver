import jax.numpy as jnp
from profiles.misc import profile_template, get_steps, time_bounds

def generate_linear_profile(mov_amp, time_grid, total_time, ksi_start, ksi_stop, eta):

    coord_profile, amp_profile = profile_template(
        mov_amp, time_grid, total_time, ksi_start, ksi_stop, eta)

    move_time, rise_time, fall_time = time_bounds(total_time, eta)
    rise, move, fall, wait = get_steps(time_grid, eta, total_time)
    distance = ksi_stop - ksi_start

    coord_profile += jnp.where(move, distance * (time_grid - rise_time) / move_time, coord_profile)

    return coord_profile, amp_profile
