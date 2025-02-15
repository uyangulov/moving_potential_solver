
from static_potential import StaticPotential
from moving_potential import MovingPotentials
from utils import K_momentum_space
import jax.numpy as jnp
from jax import jit, lax
from utils import compute_inverse_fft, compute_fft
from functools import partial
from utils import kinetic_energy, dot, dot2, compute_mean_n


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

    @staticmethod
    def step_fn(psi, time_index, coord_grid, coord_profiles, amp_profiles,
                kinetic_propagator, kinetic_term, static_potential_term, time_step, psi_target, mps: MovingPotentials):

        moving_potential_term = mps.V_moving(
            coord_grid, coord_profiles, amp_profiles, time_index)

        potential_half_propagator = jnp.exp(
            -1j * (moving_potential_term + static_potential_term) * time_step / 2)

        psi *= potential_half_propagator
        psi_momentum = compute_fft(psi, axis=-1)
        psi_momentum *= kinetic_propagator
        psi = compute_inverse_fft(psi_momentum, axis=-1)
        psi *= potential_half_propagator

        K = kinetic_energy(psi_momentum, kinetic_term)
        psi_norm = dot2(psi, psi)
        fidelity = dot2(psi, psi_target)
        
        qnum = compute_mean_n(
            psi, coord_grid, B=500, a_mt=amp_profiles[..., time_index], delta_mt=mps.width, x_mt=coord_profiles[..., time_index])
        
        stats = (K, psi_norm, fidelity, qnum)

        return psi, stats

    def prepare_kinetic(self, momentum_grid, time_step):
        kinetic_term = K_momentum_space(self.B, momentum_grid)
        kinetic_term = self.__expand_dims(kinetic_term)
        kinetic_propagator = jnp.exp(-1j * kinetic_term * time_step)
        return kinetic_term, kinetic_propagator

    def prepare_static_potential(self, coord_grid):
        static_potential_term = self.sp.V(coord_grid)
        static_potential_term = self.__expand_dims(static_potential_term)
        return static_potential_term

    def prepare_left_psi(self, coord_grid):
        psi = self.sp.left_groundstate_psi(coord_grid)
        psi = self.__reshape_to_fit_problem(psi)
        return psi

    def prepare_right_psi(self, coord_grid):
        psi = self.sp.right_groundstate_psi(coord_grid)
        psi = self.__reshape_to_fit_problem(psi)
        return psi

    def evolve(self, coord_grid, time_grid, momentum_grid, psi_start=None, psi_target=None, reverse=False):

        time_step = (time_grid[1] - time_grid[0]) / self.C
        time_indices = jnp.arange(len(time_grid))

        if reverse:
            time_step = -time_step
            time_indices = jnp.arange(len(time_grid) - 1, -1, -1)

        if psi_start is None:
            print("start psi not specified so creating one")
            psi_start = self.prepare_right_psi(
                coord_grid) if reverse else self.prepare_left_psi(coord_grid)

        if psi_target is None:
            print("final psi not specified so creating one")
            psi_target = self.prepare_left_psi(
                coord_grid) if reverse else self.prepare_right_psi(coord_grid)

        kinetic_term, kinetic_propagator = self.prepare_kinetic(
            momentum_grid, time_step)
        static_potential_term = self.prepare_static_potential(coord_grid)

        coord_profiles, amp_profiles = self.mps.populate_profiles(time_grid)

        # Create a partially applied step function
        step_fn_partial = jit(partial(
            Solver.step_fn,
            coord_grid=coord_grid,
            coord_profiles=coord_profiles,
            amp_profiles=amp_profiles,
            kinetic_propagator=kinetic_propagator,
            kinetic_term=kinetic_term,
            static_potential_term=static_potential_term,
            time_step=time_step,
            psi_target=psi_target,
            mps=self.mps
        ))

        final_psi, stats = lax.scan(step_fn_partial, psi_start, time_indices)

        return final_psi, stats

    def back_and_forth(self, coord_grid, time_grid, momentum_grid, psi_start=None):

        if psi_start is None:
            print("start psi not specified so creating one")
            psi_start = self.prepare_left_psi(coord_grid)

        x, t, p = coord_grid, time_grid, momentum_grid
        end_psi, end_stats = self.evolve(
            x, t, p, psi_start=psi_start, psi_target=psi_start, reverse=False)
        back_psi, back_stats = self.evolve(
            x, t, p, psi_start=end_psi, psi_target=psi_start, reverse=True)

        return back_psi, (jnp.concatenate((s1, s2), axis=0) for s1, s2 in zip(end_stats, back_stats))
