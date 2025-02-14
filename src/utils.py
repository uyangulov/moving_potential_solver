import jax.numpy as jnp


def second_derivative_matrix(N, step):
    coeffs_matrix = -2 * jnp.eye(N) + jnp.eye(N, k=-1) + jnp.eye(N, k=1)
    return coeffs_matrix / step**2

# kinetic_operator (square matrix) in coordinate space


def K_coordspace(borne_parameter, coord_grid):
    N = coord_grid.shape[0]
    coord_step = coord_grid[1] - coord_grid[0]
    return -1 / borne_parameter * second_derivative_matrix(N, coord_step)

# kinetic_operator diagonal in momentum space


def K_momentum_space(borne_parameter, momentum_grid):
    return 1 / borne_parameter * momentum_grid ** 2


def compute_fft(field, axis):
    return jnp.fft.fftshift(jnp.fft.fft(field, axis=axis), axes=axis)


def compute_inverse_fft(fft_field, axis):
    return jnp.fft.ifft(jnp.fft.ifftshift(fft_field, axes=axis), axis=axis)

# return grid able to represent every fourier component of state with desired energy of QM harmonic osc
# energy in units of A_st


def adaptive_grid(left, right, required_energy, B):

    coord_step = jnp.pi * jnp.sqrt(1 / (B * required_energy)) / 10
    coord_grid = jnp.arange(left, right, coord_step, dtype=jnp.complex64)
    N_x = coord_grid.shape[0]

    momentum_span = 2 * jnp.pi / coord_step
    momentum_grid = jnp.linspace(-momentum_span /
                                 2, +momentum_span / 2, N_x, dtype=jnp.complex64)
    momentum_step = (momentum_grid[1] - momentum_grid[0]).real

    return N_x, coord_grid, coord_step, momentum_grid, momentum_step


def visualize_profiles(time_grid, coord_profiles, amp_profiles, total_times, profile_kind_to_index, amplitudes, ax):

    amplitude_index = 1

    for profile, profile_index in profile_kind_to_index.items():

        coord_prof = coord_profiles[profile_index, amplitude_index]
        amp_prof = amp_profiles[profile_index, amplitude_index]

        for total_time_index, total_time in enumerate(total_times):

            ax[profile_index, 0].plot(
                time_grid, coord_prof[total_time_index].real, label=f'Time: {total_time:.2f}')

            ax[profile_index, 1].plot(
                time_grid, amp_prof[total_time_index].real, label=f'Time: {total_time:.2f}')

        ax[profile_index, 0].set_title(f'{profile} Coordinate Profile')
        ax[profile_index, 0].set_ylabel('Coordinate')

        ax[profile_index, 1].set_title(f'{profile} Amplitude Profile')
        ax[profile_index, 1].set_ylabel('Amplitude')

    # Set common xlabel
    for ax_row in ax:
        ax_row[0].set_xlabel('Time')
        ax_row[1].set_xlabel('Time')
        ax_row[1].set_ylim(-1, amplitudes[amplitude_index]+1)


def dot(x, y):
    return jnp.vecdot(x, y, axis=-1)

def dot2(x, y):
    return jnp.abs(dot(x, y))**2

def fid(x, y):
    return dot2(x / jnp.sqrt(dot(x,x)), y / jnp.sqrt(dot(y,y)))

def kinetic_energy(psi_momentum, kinetic_term):
    return dot(psi_momentum * kinetic_term, psi_momentum) / (1e-6 + norm(psi_momentum))

def potential_energy(psi, potential_term):
    return dot(psi * potential_term, psi) / (1e-6 + norm(psi))

def compute_mean_energy(psi, psi_momentum, potential_term, kinetic_term):
    return kinetic_energy(psi_momentum, kinetic_term) + potential_energy(psi, potential_term)
