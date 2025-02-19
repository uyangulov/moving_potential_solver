import jax.numpy as jnp
from profiles.misc import profile_template, get_steps, time_bounds, magic_poly


def generate_minjerk_profile(mov_amp, time_grid, total_time, ksi_start, ksi_stop, eta):

    distance = ksi_stop - ksi_start

    coord_profile, amp_profile = profile_template(
        mov_amp, time_grid, total_time, distance, eta)

    move_time, rise_time, fall_time = time_bounds(total_time, eta)
    rise, move, fall, wait = get_steps(time_grid, eta, total_time)

    coord_profile += jnp.where(
        move,
        distance * magic_poly((time_grid - rise_time) / move_time),
        0
    )
    coord_profile += ksi_start
    return coord_profile, amp_profile
