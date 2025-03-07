from profiles.linear_profile import generate_linear_profile
from profiles.minjerk_profile import generate_minjerk_profile
from profiles.hybrid_profile import generate_hybrid_profile
from profiles.deepen_only import generate_deepen_only

# from profiles import generate_sta_profile


def generate_profile(profile, amp, time_grid, total_time, x_left, x_right, eta, *args):
    """Generates coordinate and amplitude profiles based on the profile type."""
    if profile == "Linear":
        return generate_linear_profile(amp, time_grid, total_time, x_left, x_right, eta)
    
    elif profile == "Minjerk":
        return generate_minjerk_profile(amp, time_grid, total_time, x_left, x_right, eta)
    elif profile == "Hybrid":
        return generate_hybrid_profile(amp, time_grid, total_time, x_left, x_right, eta, hybr=0.4)
    elif profile == "Deepen":
        return generate_deepen_only(amp, time_grid, total_time, x_left, eta)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")
