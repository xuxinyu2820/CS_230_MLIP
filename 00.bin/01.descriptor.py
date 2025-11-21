import jax.numpy as jnp

def compute_rij(
    positions: jnp.ndarray,
    i: int,
    j: int,
    use_pbc: bool = False,
    box: jnp.ndarray | None = None,
):
    """Relative displacement r_ij = R_i - R_j, with optional PBC."""
    Ri = positions[i]
    Rj = positions[j]
    r = Ri - Rj
    d = jnp.linalg.norm(r)

    if not use_pbc:
        return r, d

    # box: (3,)
    L = jnp.asarray(box)  # (3,)
    r_pbc = r - L * jnp.round(r / L)
    d_pbc = jnp.linalg.norm(r_pbc)
    return r_pbc, d_pbc

def smooth_cutoff(r: jnp.ndarray,
                  r_cs: float,
                  r_c: float,
                  eps: float = 1e-8) -> jnp.ndarray:
    """Smooth 1/r -> 0 between r_cs and r_c (DeepPot-style)."""
    # avoid 1/0
    r_safe = jnp.maximum(r, eps)
    inv_r = 1.0 / r_safe

    # polynomial part for r_cs <= r < r_c
    u = (r_safe - r_cs) / (r_c - r_cs)
    poly = u**3 * (-6.0 * u**2 + 15.0 * u - 10.0)

    s_mid = inv_r * (poly + 1.0)
    s_zero = jnp.zeros_like(r_safe)

    # piecewise
    s = jnp.where(
        r_safe < r_cs,
        inv_r,
        jnp.where(r_safe < r_c, s_mid, s_zero),
    )
    return s


def build_feat(
    rij_vec: jnp.ndarray,  # (3,)
    s: jnp.ndarray,        # ()
) -> jnp.ndarray:
    """Make (s, s*x, s*y, s*z) from s and relative coord."""
    sx = s * rij_vec[0]
    sy = s * rij_vec[1]
    sz = s * rij_vec[2]
    return jnp.stack([s, sx, sy, sz], axis=0)

# def compute_dp_norm_stats(
#     feats4: jnp.ndarray,
#     eps: float = 1e-8,
# ):
#     """
#     Compute global stats for DP-style 4-dim neighbor features.
#     feats4: (..., 4) = (s, sx, sy, sz)
#     """
#     # first channel stats
#     s = feats4[..., 0]
#     s_mean = jnp.mean(s)
#     s_std = jnp.std(s) + eps

#     # coord channels: stack (sx, sy, sz) together
#     coords = feats4[..., 1:]           # (..., 3)
#     # one shared std over x/y/z
#     coord_std = jnp.sqrt(jnp.mean(coords * coords)) + eps

#     return s_mean, s_std, coord_std

def norm_feature(
    feat4: jnp.ndarray,  # (4,) = (s, sx, sy, sz)
    s_mean: float,
    s_std: float,
    coord_std: float,
) -> jnp.ndarray:
    """Normalize DP 4-dim feature."""
    s  = feat4[0]
    sx = feat4[1]
    sy = feat4[2]
    sz = feat4[3]

    s_hat  = (s - s_mean) / s_std
    sx_hat = sx / coord_std
    sy_hat = sy / coord_std
    sz_hat = sz / coord_std

    return jnp.stack([s_hat, sx_hat, sy_hat, sz_hat], axis=0)

def build_env_matrix(
    i: int,
    positions: jnp.ndarray,     # (N, 3)
    neighbors_i: jnp.ndarray,   # (K,)
    max_sel: int,               # e.g. 32
    r_cs: float,
    r_c: float,
    stats: dict,                # {"s_mean": .., "s_std": .., "coord_std": ..}
    use_pbc: bool = False,
    box: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    For center atom i, take closest `max_sel` neighbors (any species).
    No vmap: do per (i,j) in Python.
    If actual neighbors < max_sel, pad with zeros.
    Output: (max_sel, 4)
    """
    s_mean = stats["s_mean"]
    s_std = stats["s_std"]
    coord_std = stats["coord_std"]

    feats = []   # to collect (4,)
    dists = []   # to collect scalar distance

    # ensure array
    neighbors_i = jnp.asarray(neighbors_i)

    for j_idx in neighbors_i:
        j = int(j_idx)

        # 1) geometry
        rij_vec, rij = compute_rij(
            positions,
            i,
            j,
            use_pbc=use_pbc,
            box=box,
        )

        # 2) smooth weight
        s_ij = smooth_cutoff(rij, r_cs, r_c)

        # 3) raw 4-d
        raw_feat = build_feat(rij_vec, s_ij)

        # 4) lookup stats → normalize
        norm_feat = norm_feature(
            raw_feat,
            s_mean=s_mean,
            s_std=s_std,
            coord_std=coord_std,
        )

        feats.append(norm_feat)
        dists.append(rij)

    if len(feats) == 0:
        # just pad zeros
        return jnp.zeros((max_sel, 4), dtype=positions.dtype)

    feats = jnp.stack(feats, axis=0)   # (K, 4)
    dists = jnp.stack(dists, axis=0)   # (K,)
    order = jnp.argsort(dists)         # (K,)
    feats_sorted = feats[order]        # (K, 4)

    K = feats_sorted.shape[0]
    take = int(min(K, max_sel))
    taken = feats_sorted[:take]        # (take, 4)

    if take < max_sel:
        pad = jnp.zeros((max_sel - take, 4), dtype=positions.dtype)
        taken = jnp.concatenate([taken, pad], axis=0)

    return taken   # (max_sel, 4)



