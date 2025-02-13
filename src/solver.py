
from static_potential import StaticPotential
from moving_potential import MovingPotentials
from utils import K_momentum_space
import jax.numpy as jnp
from jax import jit, lax
from utils import compute_inverse_fft, compute_fft


class Solver:

    def __init__(self, sp: StaticPotential, mps: MovingPotentials):

        self.sp = sp
        self.mps = mps

        self.B = sp.borne_parameter
        self.C = sp.c_prefactor

        self.shape = (mps.n_profiles, mps.n_amps, mps.n_total_times)

    def __expand_dims(self, obj):
        return obj[None, None, None, :]

    def __reshape_to_fit_problem(self, obj):
        return self.__expand_dims(obj) + jnp.zeros(self.shape)[..., jnp.newaxis]

    # (mps.n_profiles, mps.n_amps, mps.n_total_times, N_x)
    def init_psi_0(self, coord_grid):

        psi_0, _, _ = self.sp.groundstate(coord_grid)
        psi_0 = self.__reshape_to_fit_problem(psi_0)
        #print(f"Initialized psi_0 of shape {psi_0.shape}")
        return psi_0

    def final_groundstate(self, coord_grid):
        _, psi_f, _ = self.sp.groundstate(coord_grid)
        psi_f = self.__reshape_to_fit_problem(psi_f)
        #print(f"Initialized psi_f of shape {psi_f.shape}")
        return psi_f

    def init_operators(self, coord_grid, time_grid, momentum_grid):

        time_step = time_grid[1] - time_grid[0]

        kinetic_term = K_momentum_space(self.B, momentum_grid) / self.C
        kinetic_term = self.__expand_dims(kinetic_term)
        kinetic_propagator = jnp.exp(-1j * kinetic_term * time_step)
        #print(f"kinetic_propagator shape {kinetic_propagator.shape}")

        static_potential_term = self.sp.V(coord_grid) / self.C
        static_potential_term = self.__expand_dims(static_potential_term)
        #print(f"static_potential_term shape {static_potential_term.shape}")

        return kinetic_term, kinetic_propagator, static_potential_term

    def solve(self, coord_grid, time_grid, momentum_grid):

        psi_0 = self.init_psi_0(coord_grid)
        time_step = time_grid[1] - time_grid[0]
        kinetic_term, kinetic_propagator, static_potential_term = self.init_operators(
            coord_grid, time_grid, momentum_grid)
        
        coord_profiles, amp_profiles = self.mps.populate_profiles(time_grid)
        #print(f"Populated profiles")

        psi_f = self.final_groundstate(coord_grid)

        @jit
        def __step_fn(psi, time_index):

            moving_potential_term = self.mps.V_moving(
                coord_grid, coord_profiles, amp_profiles, time_index) / self.C
            potential_half_propagator = jnp.exp(
                -1j * (moving_potential_term + static_potential_term) * time_step / 2)

            psi *= potential_half_propagator
            psi = compute_fft(psi, axis=-1)
            psi *= kinetic_propagator
            kinetic = self.__dot(psi * kinetic_term, psi)
            psi = compute_inverse_fft(psi, axis=-1)
            psi *= potential_half_propagator

            return psi, (self.__dot2(psi, psi), kinetic, self.__fid(psi, psi_f))

        # Start from psi_0, iterate over time_grid (excluding the first time)
        final_psi, stats = lax.scan(
            __step_fn, psi_0, jnp.arange(len(time_grid)))

        return final_psi, stats  # Return the final wavefunction at the last timestep

    def __dot(self, x, y):
        return jnp.vecdot(x, y, axis=-1)

    def __dot2(self, x, y):
        return jnp.abs(self.__dot(x, y))**2

    def __norm(self, x):
        return self.__dot2(x, x)

    def __fid(self, x, y):
        return self.__dot2(x, y) / (self.__norm(x) * self.__norm(y))
