import jax.numpy as jnp
import json
from utils import K_coordspace


class StaticPotential:

    def __init__(self, x_left, x_right, borne_parameter, c_prefactor):

        # left and right static trap coords
        self.x_left = x_left
        self.x_right = x_right

        # some magic constants
        self.borne_parameter = borne_parameter
        self.c_prefactor = c_prefactor

    # return static_potential_curve
    def V(self, coord_grid):
        first = -jnp.exp(-(coord_grid - self.x_left) ** 2 / 2)
        second = -jnp.exp(-(coord_grid - self.x_right) ** 2 / 2)
        # first = -1 + (coord_grid - self.x_left) ** 2 / 2
        # second = -1 + (coord_grid - self.x_right) ** 2 / 2
        return first + second

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as file:
            params = json.load(file)

        h_reduced = params["reduced_planck_constant"]
        A_st = params["static_tweezer_amplitude"]
        sigma_st = params["static_beam_width"]
        d = params["trap_distance"]
        m = params["atom_mass"]

        # Compute derived parameters
        B = A_st * (sigma_st ** 2) * 2 * m / (h_reduced ** 2)
        w_st = jnp.sqrt(A_st / m) / sigma_st
        t_st = 2 * jnp.pi / w_st
        C = h_reduced / (A_st * t_st)
        x_right = d / sigma_st

        return cls(x_left=0, x_right=x_right, borne_parameter=B, c_prefactor=C)

    def __repr__(self):
        return (
            f"StaticPotential(\n"
            f"    x_left={self.x_left:.3f},\n"
            f"    x_right={self.x_right:.3f},\n"
            f"    borne_parameter={self.borne_parameter:.3e},\n"
            f"    c_prefactor={self.c_prefactor:.3e}\n"
            f")"
        )

    # eigenstate right before evolution
    def groundstate_psis(self, coord_grid):
        alpha = (self.borne_parameter / 2) ** 0.25
        left_eigvec =  jnp.exp(-(alpha * (coord_grid - self.x_left))**2)
        right_eigvec = jnp.exp(-(alpha * (coord_grid - self.x_right))**2)
        return left_eigvec, right_eigvec

    def left_groundstate_psi(self, coord_grid):
        return self.groundstate_psis(coord_grid)[0]

    def right_groundstate_psi(self, coord_grid):
        return self.groundstate_psis(coord_grid)[1]

    def groundstate_energy(self):
        eigval = jnp.sqrt(2 / self.borne_parameter) / 2
        return eigval

