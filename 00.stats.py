import json
from pathlib import Path
import jax.numpy as jnp
from descriptor import *

def accumulate_stats_for_frame(
    positions: jnp.ndarray,          # (N, 3)
    neighbor_list,                   # list/array, len = N, each is (K_i,)
    r_cs: float,
    r_c: float,
    use_pbc: bool = False,
    box: jnp.ndarray | None = None,
):
    """Scan one frame and return all (s, sx, sy, sz) stacked."""
    feats = []

    N = positions.shape[0]
    for i in range(N):
        neigh_i = neighbor_list[i]
        # make sure it's jnp
        neigh_i = jnp.asarray(neigh_i)
        for j_idx in neigh_i:
            j = int(j_idx)

            # 1) rij, |rij|
            rij_vec, rij = compute_rij(
                positions,
                i,
                j,
                use_pbc=use_pbc,
                box=box,
            )

            # 2) smooth cutoff
            s_ij = smooth_cutoff(rij, r_cs, r_c)

            # 3) build 4-d feature (not normalized yet)
            feat4 = build_feat(rij_vec, s_ij)      # (4,)

            feats.append(feat4)

    if len(feats) == 0:
        return jnp.zeros((0, 4))

    feats = jnp.stack(feats, axis=0)               # (M, 4)
    return feats


def compute_global_stats(
    all_feats: jnp.ndarray,
    eps: float = 1e-8,
):
    """Compute s_mean, s_std, coord_std from stacked feats."""
    # channel 0: s
    s = all_feats[:, 0]
    s_mean = jnp.mean(s)
    s_std = jnp.std(s) + eps

    # channels 1..3: coords
    coords = all_feats[:, 1:]              # (M, 3)
    coord_std = jnp.sqrt(jnp.mean(coords * coords)) + eps

    return float(s_mean), float(s_std), float(coord_std)


def main():
    # ====== you need to modify this part to load your data ======
    # Here I assume:
    # - you have a list of frames
    # - each frame gives: positions, neighbor_list, box
    # You can replace this with your own loader.
    #
    # Example placeholder:
    dataset = []  # fill with your data

    # e.g. dataset = [
    #   {"positions": jnp.array(...), "neighbors": [...], "box": jnp.array([Lx, Ly, Lz])},
    #   ...
    # ]

    r_cs = 5.5
    r_c = 6.0
    use_pbc = True

    all_feats_list = []

    for frame in dataset:
        pos = frame["positions"]
        nbh = frame["neighbors"]
        box = frame.get("box", None)

        feats_f = accumulate_stats_for_frame(
            pos,
            nbh,
            r_cs=r_cs,
            r_c=r_c,
            use_pbc=use_pbc,
            box=box,
        )
        if feats_f.shape[0] > 0:
            all_feats_list.append(feats_f)

    if len(all_feats_list) == 0:
        raise RuntimeError("No neighbor features collected. Check your data loader.")

    all_feats = jnp.concatenate(all_feats_list, axis=0)   # (total_M, 4)

    s_mean, s_std, coord_std = compute_global_stats(all_feats)

    stats = {
        "r_cs": float(r_cs),
        "r_c": float(r_c),
        "s_mean": s_mean,
        "s_std": s_std,
        "coord_std": coord_std,
        "num_samples": int(all_feats.shape[0]),
    }

    out_path = Path("stats.json")
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[info] wrote stats to {out_path.resolve()}")


if __name__ == "__main__":
    main()
