import jax.numpy as jnp
from profiles.misc import profile_template, get_steps, time_bounds, magic_poly


def hybrid_normalized(move_time, time_grid, hybr):

    condition1 = 2 * time_grid < (1 - hybr) * move_time
    condition2 = ((1 - hybr) * move_time <= 2 *
                  time_grid) & (2 * time_grid < (1 + hybr) * move_time)
    condition3 = ((1 + hybr) * move_time <= 2 * time_grid)

    denominator = 8 + 7 * hybr
    frac1 = 8 * (1 - hybr) / denominator
    frac2 = 15 / denominator
    frac3 = 7 * (1 - hybr) / denominator

    case1 = frac1 * magic_poly(time_grid / (1 - hybr) / move_time)
    case2 = frac2 * time_grid / move_time - frac3 / 2
    case3 = frac1 * \
        magic_poly((time_grid - move_time * hybr) /
                   (1-hybr) / move_time) + frac2 * hybr

    result = jnp.where(condition1, case1, 0)
    result = jnp.where(condition2, case2, result)
    result = jnp.where(condition3, case3, result)

    return result


def generate_hybrid_profile(mov_amp, time_grid, total_time, ksi_start, ksi_stop, eta, hybr):

    distance = ksi_stop - ksi_start

    coord_profile, amp_profile = profile_template(
        mov_amp, time_grid, total_time, distance, eta)

    move_time, rise_time, fall_time = time_bounds(total_time, eta)
    rise, move, fall, wait = get_steps(time_grid, eta, total_time)

    coord_profile += jnp.where(
        move,
        distance *
        hybrid_normalized(move_time, time_grid - (1 - eta)
                          * total_time / 3, hybr),
        0
    )
    coord_profile += ksi_start

    return coord_profile, amp_profile
