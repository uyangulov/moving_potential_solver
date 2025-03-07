import jax.numpy as jnp
from profiles.generate import generate_profile

class MovingPotentials:

    def __init__(self, eta, width, amps, total_times, src, dest):
        self.width = width
        self.eta = eta
        self.src = src
        self.dest = dest

        self.amps = amps
        self.total_times = total_times

        #self.profile_kinds = ["Linear", "Minjerk", "Hybrid"]  # , "STA"]
        self.profile_kinds = ["Deepen"]

    @property
    def n_total_times(self):
        return len(self.total_times)

    @property
    def n_amps(self):
        return len(self.amps)

    @property
    def n_profiles(self):
        return len(self.profile_kinds)

    @property
    def profile_kind_to_index(self):
        return {string: index for index, string in enumerate(self.profile_kinds)}

    def populate_profiles(self, time_grid):
        """
        Generates coordinate and amplitude profiles for different parameter configurations.

        Args:
            time_grid (jax.numpy.ndarray): 1D array of time points.

        Returns:
            tuple:
                - coord_profiles (jax.numpy.ndarray): Coordinate profiles of shape 
                  `(n_profiles, n_amps, n_total_times, N_t)`, where `N_t` is the length of `time_grid`.
                - amp_profiles (jax.numpy.ndarray): Amplitude profiles of shape 
                  `(n_profiles, n_amps, n_total_times, N_t)`.
        """
        N_t = len(time_grid)
        shape = (self.n_profiles, self.n_amps, self.n_total_times, N_t)

        amp_profiles = jnp.zeros(shape=shape, dtype=jnp.complex64)
        coord_profiles = jnp.zeros(shape=shape, dtype=jnp.complex64)

        for profile, profile_index in self.profile_kind_to_index.items():
            for amp_index, amp in enumerate(self.amps):
                for total_time_index, total_time in enumerate(self.total_times):
                    try:
                        coord_prof, amp_prof = generate_profile(
                            profile, amp, time_grid, total_time, self.src, self.dest, self.eta, self.width
                        )
                        coord_profiles = coord_profiles.at[profile_index, amp_index, total_time_index].set(coord_prof)
                        amp_profiles = amp_profiles.at[profile_index, amp_index, total_time_index].set(amp_prof)
                    except ValueError as e:
                        print(f"Skipping profile {profile}: {e}")

        return coord_profiles, amp_profiles

    def V_moving(self, coord_grid, coord_profiles, amp_profiles, time_index):
        """
        Computes the potential at a given time index.

        Args:
            coord_grid (jax.numpy.ndarray): 1D array of spatial coordinate points.
            coord_profiles (jax.numpy.ndarray): Coordinate profiles of shape `(n_profiles, n_amps, n_total_times, N_t)`.
            amp_profiles (jax.numpy.ndarray): Amplitude profiles of shape `(n_profiles, n_amps, n_total_times, N_t)`.
            time_index (int): The time index to evaluate the potential.

        Returns:
            jax.numpy.ndarray: Potential values of shape `(n_profiles, n_amps, n_total_times, N_x)`, 
            where `N_x` is the length of `coord_grid`.
        """
        coord_profile_this_time = coord_profiles[..., time_index]
        amp_profile_this_time = amp_profiles[..., time_index]

        # Shape (n_profiles, n_amps, n_total_times, 1)
        coord_profile_this_time_expanded = coord_profile_this_time[..., jnp.newaxis]
        amp_profile_this_time_expanded = amp_profile_this_time[..., jnp.newaxis]

        # Shape (1,1,1,N_x) - (n_profiles, n_amps, n_total_times, 1) = (n_profiles, n_amps, n_total_times, N_x)
        centered = coord_grid[None, None, None, :] - coord_profile_this_time_expanded

        return -amp_profile_this_time_expanded * jnp.exp(-centered**2 / (2 * self.width**2))
        #return amp_profile_this_time_expanded * (-1 + (centered**2) / (2 * self.width**2))

    def __repr__(self):
        """
        Returns a string representation of the MovingPotentials object.

        Returns:
            str: Formatted string with class attributes.
        """
        return (
            f"MovingPotentials(\n"
            f"    eta={self.eta:.3e},\n"
            f"    width={self.width:.3e},\n"
            f"    src={self.src:.3f},\n"
            f"    dest={self.dest:.3f},\n"
            f"    n_amps={self.n_amps},\n"
            f"    n_total_times={self.n_total_times},\n"
            f"    n_profiles={self.n_profiles},\n"
            f"    profile_kinds={self.profile_kinds}\n"
            f")"
        )
