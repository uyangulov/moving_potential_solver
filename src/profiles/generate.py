from profiles import generate_linear_profile
from profiles import generate_minjerk_profile
from profiles import generate_sta_profile

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