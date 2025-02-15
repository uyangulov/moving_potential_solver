# import pytest
# import numpy as np
# from scipy.special import hermite
# import jax.numpy as jnp
# from src.utils import compute_fft

# # H = A_st(-1/B * d^2/dx^2 + 1/2 * D * (xi)**2
# # psi(x) ~ exp()


# def harmonic_oscillator_wavefunction(n, grid, D, B):
#     """
#     Compute the n-th quantum harmonic oscillator wavefunction in coord space
#     """
#     hermite_poly = hermite(n)
#     alpha = (D * B / 2) ** 0.25
#     transformed = alpha * grid
#     return alpha * jnp.exp(-0.5 * transformed ** 2) * hermite_poly(transformed)


# def harmonic_oscillator_momentum_wavefunction(n, grid, D, B):
#     """
#     Compute the n-th quantum harmonic oscillator wavefunction in momentum space
#     """
#     hermite_poly = hermite(n)
#     beta = (2 / (D * B)) ** 0.25
#     transformed = beta * grid
#     return beta * jnp.exp(-0.5 * transformed ** 2) * hermite_poly(transformed)


# class TestFourier:
#     @pytest.mark.parametrize("n", [0, 1, 2, 3])
#     def test_harmonic_oscillator_fft(self, n):
#         """Test Fourier transform of harmonic oscillator wavefunctions against exact results."""

#         D, B, N = 1, 500, 1000
#         coord_grid = jnp.linspace(-10, 10, N)
#         coord_step = coord_grid[1] - coord_grid[0]
#         psi = harmonic_oscillator_wavefunction(n, coord_grid, D, B)

#         momentum_span = 2 * jnp.pi / coord_step
#         momentum_grid = jnp.linspace(-momentum_span /
#                                     2, +momentum_span / 2, N, dtype=jnp.complex64)
#         momentum_step = (momentum_grid[1] - momentum_grid[0]).real

#         fft_result = compute_fft(psi, axis=-1)

#         # Normalize FFT result correctly
#         psi_p_fft = np.fft.fftshift(fft_result)

#         # Compute exact p-space wavefunction
#         psi_p_exact = exact_momentum_space_wavefunction(n, p, m, omega)

#         # Compare results
#         assert np.allclose(np.abs(psi_p_fft), np.abs(psi_p_exact), atol=1e-2)
