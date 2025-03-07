import jax.numpy as jnp
from profiles.misc import get_steps, time_bounds, magic_poly

#rise then hold
def generate_deepen_only(mov_amp, time_grid, total_time, coord, eta):

    coord_profile = jnp.full_like(a=time_grid, fill_value=coord)
    amp_profile = jnp.zeros_like(a=time_grid)

    rise = time_grid < eta * total_time
    hold = time_grid >= eta * total_time
    
    # Generate amplitude profile
    amp_profile = jnp.where(rise, mov_amp * time_grid / total_time / eta, amp_profile) 
    amp_profile = jnp.where(hold, mov_amp, amp_profile)

    return coord_profile, amp_profile
