# 03.Model.py
# Fully connected per-atom MLP in JAX/nnx.
# - Input:  (N, M) features per structure (N atoms, M features per atom)
# - Output: per-atom energies E_i (shape [N]) and total energy E (scalar)
# - Forces: F = -grad_R E, computed if you pass a differentiable descriptor_fn(R, ...)
# - Virial: helper from positions R and forces F with selectable convention
#
# Usage examples:
#   python 03.Model.py --in-dim 128 --hidden 240 240 240 --act tanh
#   # (demo prints shapes; wire `forces()`/`virial()` once you have descriptor_fn)
#
# Notes:
# - Keep inputs as float32 for speed on accelerators.
# - Activation default is tanh (common in DP-style fitting nets).
# - Hidden sizes default to [240, 240, 240]; change via CLI or constructor.

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.experimental import nnx as nnx


# -----------------------------
# Utilities
# -----------------------------

_ACTS = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "softplus": jax.nn.softplus,
}

def _get_act(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name not in _ACTS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTS)}.")
    return _ACTS[name]


# -----------------------------
# Model
# -----------------------------

class PerAtomMLP(nnx.Module):
    """Simple per-atom MLP: (N, M) -> (N,) energy_i

    Hidden sizes and activation are configurable. Final layer is linear.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        act: str = "tanh",
        *,
        rngs: nnx.Rngs,
    ):
        self.act_name = act
        dims = [in_dim] + list(hidden) + [1]
        # Build Linear layers
        self.layers: List[nnx.Linear] = []
        for din, dout in zip(dims[:-1], dims[1:]):
            self.layers.append(nnx.Linear(din, dout, rngs=rngs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (N, M) -> returns per-atom energy (N,)"""
        act = _get_act(self.act_name)
        h = x
        L = len(self.layers)
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < L - 1:  # no activation on final layer
                h = act(h)
        return jnp.squeeze(h, axis=-1)  # (N,)


# -----------------------------
# High-level energy / forces / virial API
# -----------------------------

@dataclass
class FittingModel:
    """Container with the MLP and convenience fns."""

    mlp: PerAtomMLP
    act: str
    hidden: Sequence[int]
    in_dim: int

    # ---------- Feature-mode (features already computed) ----------
    @staticmethod
    def per_atom_energy_from_features(mlp: PerAtomMLP, x_nm: jnp.ndarray) -> jnp.ndarray:
        """Compute per-atom energies given features x_nm with shape (N, M)."""
        return mlp(x_nm)

    @staticmethod
    def total_energy_from_features(mlp: PerAtomMLP, x_nm: jnp.ndarray) -> jnp.ndarray:
        """Sum of per-atom energies."""
        return jnp.sum(mlp(x_nm))

    # ---------- Position-mode (descriptor_fn supplies features from positions) ----------
    @staticmethod
    def total_energy_from_positions(
        mlp: PerAtomMLP,
        R: jnp.ndarray,               # (N, 3)
        descriptor_fn: Callable[..., jnp.ndarray],  # descriptor_fn(R, *args, **kwargs) -> (N, M) (differentiable in R)
        *desc_args,
        **desc_kwargs,
    ) -> jnp.ndarray:
        """Total energy E(R) given a differentiable descriptor_fn."""
        x_nm = descriptor_fn(R, *desc_args, **desc_kwargs)  # (N, M)
        return jnp.sum(mlp(x_nm))

    @staticmethod
    def forces_from_positions(
        mlp: PerAtomMLP,
        R: jnp.ndarray,               # (N, 3)
        descriptor_fn: Callable[..., jnp.ndarray],
        *desc_args,
        **desc_kwargs,
    ) -> jnp.ndarray:
        """Forces F(R) = -∂E/∂R. Requires descriptor_fn to be JAX-differentiable w.r.t. R."""
        def energy_wrt_R(R_in):
            return FittingModel.total_energy_from_positions(mlp, R_in, descriptor_fn, *desc_args, **desc_kwargs)
        dE_dR = grad(energy_wrt_R)(R)
        return -dE_dR  # (N, 3)

    @staticmethod
    def virial_from_R_F(
        R: jnp.ndarray,  # (N, 3)
        F: jnp.ndarray,  # (N, 3)
        convention: str = "atomic",
    ) -> jnp.ndarray:
        """Compute a virial/stress-like tensor from positions and forces.

        Args
        ----
        convention:
          - "atomic":  W = - sum_i (R_i ⊗ F_i)
          - "half":    W = - 0.5 * sum_i (R_i ⊗ F_i)    # sometimes used in literature

        Returns
        -------
        (3, 3) virial-like tensor. Divide by volume for Cauchy stress if needed.
        """
        if convention not in ("atomic", "half"):
            raise ValueError("convention must be 'atomic' or 'half'")
        coef = -1.0 if convention == "atomic" else -0.5
        return coef * jnp.einsum("ni,nj->ij", R, F)


# -----------------------------
# Factory & CLI
# -----------------------------

def build_model(
    in_dim: int,
    hidden: Sequence[int] = (240, 240, 240),
    act: str = "tanh",
    seed: int = 0,
) -> FittingModel:
    rngs = nnx.Rngs(jax.random.PRNGKey(seed))
    mlp = PerAtomMLP(in_dim, hidden, act=act, rngs=rngs)
    return FittingModel(mlp=mlp, act=act, hidden=tuple(hidden), in_dim=in_dim)


def _demo(args):
    # Build model
    model = build_model(args.in_dim, tuple(args.hidden), args.act, args.seed)

    # Demo forward on random features (no positions)
    key = jax.random.PRNGKey(args.seed + 1)
    N = args.n_atoms
    x = jax.random.normal(key, (N, args.in_dim), dtype=jnp.float32)
    Ei = FittingModel.per_atom_energy_from_features(model.mlp, x)     # (N,)
    E  = FittingModel.total_energy_from_features(model.mlp, x)        # ()
    print(f"[demo] per-atom energies: shape={Ei.shape}, total energy: shape={E.shape}, value={float(E):.6f}")

    # If you have a differentiable descriptor_fn(R)->(N,M), uncomment and test forces/virial:
    # def descriptor_fn(R):
    #     # Example placeholder: linear projection from positions to features
    #     # Replace with your real descriptor (must be differentiable in R)
    #     W = jax.random.normal(jax.random.PRNGKey(123), (3, args.in_dim), dtype=jnp.float32)
    #     return R @ W  # (N, M)
    #
    # R = jax.random.normal(key, (N, 3), dtype=jnp.float32)
    # F = FittingModel.forces_from_positions(model.mlp, R, descriptor_fn)  # (N, 3)
    # W = FittingModel.virial_from_R_F(R, F, convention="atomic")          # (3, 3)
    # print(f"[demo] forces shape={F.shape}, virial shape={W.shape}")

def parse_args():
    p = argparse.ArgumentParser(description="Fully-connected per-atom MLP in JAX/nnx")
    p.add_argument("--in-dim", type=int, required=True, help="M (features per atom)")
    p.add_argument("--hidden", type=int, nargs="+", default=[240, 240, 240],
                   help="Hidden layer sizes, e.g. --hidden 240 240 240")
    p.add_argument("--act", type=str, default="tanh", choices=list(_ACTS.keys()),
                   help="Activation function")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed")
    p.add_argument("--n-atoms", type=int, default=32, help="Demo: number of atoms N")
    return p.parse_args()

if __name__ == "__main__":
    _demo(parse_args())
