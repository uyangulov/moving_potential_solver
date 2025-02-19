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
