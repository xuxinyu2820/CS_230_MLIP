import jax
import jax.numpy as jnp

def emb_apply_row(x: jnp.ndarray, params) -> jnp.ndarray:
    """MLP: (4,) -> (emb_dim,)"""
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(W @ h + b)
    W_last, b_last = params[-1]
    y = W_last @ h + b_last
    return y

def emb_init(
    key: jax.Array,
    in_dim: int = 4,
    hidden: tuple[int, ...] = (16, 16),
    out_dim: int = 16,
):
    """Init MLP params as list[(W,b), ...]."""
    params = []
    k = key
    dims = (in_dim,) + hidden + (out_dim,)
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        k, sub = jax.random.split(k)
        W = jax.random.normal(sub, (d_out, d_in)) / jnp.sqrt(d_in)
        b = jnp.zeros((d_out,))
        params.append((W, b))
    return params

def emb_apply_env(
    env_mat: jnp.ndarray,   # (max_sel, 4)
    params,
) -> jnp.ndarray:
    """Apply row MLP to all rows."""
    apply_row = lambda row: emb_apply_row(row, params)
    return jax.vmap(apply_row)(env_mat)   # (max_sel, emb_dim)

def build_S(
    G_i: jnp.ndarray,       # (max_sel, emb_dim)
    R_i: jnp.ndarray,       # (max_sel, 4)
) -> jnp.ndarray:
    """S^i = (G^i)^T R^i -> (emb_dim, 4)."""
    return G_i.T @ R_i

def build_descriptor(
    S_i: jnp.ndarray,       # (emb_dim, 4)
    M_prime: int,
) -> jnp.ndarray:
    """D^i = S^i (S^{i<})^T, shape (emb_dim, M')."""
    S_cut = S_i[:M_prime, :]           # (M', 4)
    D_i = S_i @ S_cut.T                # (emb_dim, M')
    return D_i

def build_descriptor_flat(
    S_i: jnp.ndarray,       # (emb_dim, 4)
    M_prime: int,
) -> jnp.ndarray:
    """
        D_flat  : (emb_dim * M_prime,)
    """
    S_cut = S_i[:M_prime, :]           # (M', 4)
    D_i = S_i @ S_cut.T                # (emb_dim, M')
    D_flat = D_i.reshape(-1,)          # (emb_dim*M_prime,)
    return D_flat