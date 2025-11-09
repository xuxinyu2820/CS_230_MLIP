from typing import Sequence

import torch
from torch_geometric.data import Data, InMemoryDataset

import jax
from jax import vmap
import jax.numpy as jnp
import flax.nnx as nnx

from deepx.data.from_dft import AbInitDataSetFeature
from deepx.data.graphset import GraphSetInMemory, GraphSetOnDisk
from deepx.data.dataloader import get_loader_training

torch.set_default_dtype(torch.float32)

dft_data_feature = AbInitDataSetFeature("./dft")
graph = GraphSetInMemory(
    graph_pt_path="MD17_test.train-HE.train-FF.pt",
    dataset_name="MD17" + "_test",
    graph_type="train-FF",
    graph_save_dir="./",
    parallel_num=2,
    dft_data_feature=dft_data_feature.get_features(),
    forcibly_rebuild=True,
)
print(graph.info)
print("********************************")
print(graph[0])
random_stream = jax.random.PRNGKey(0)
train_loader, val_loader, test_loader = get_loader_training(
    graph, train_size=2, val_size=0, test_size=0, batch_size=2, graph_key=random_stream
)
batch = next(iter(train_loader))

atomic_numbers = jnp.array(batch.x)
edge_index = jnp.array(batch.edge_index)
edge_attr = jnp.array(batch.edge_attr)

def topM_edges(edge_index, edge_attr, M):
    u, v = edge_index
    N = int(jnp.max(edge_index)) + 1
    lens = edge_attr[:, 0]

    def topM_per_node(i):
        mask  = u == i
        d     = jnp.where(mask, lens, jnp.inf)
        order = jnp.argsort(d)[:M+1]
        valid = jnp.isfinite(d[order])
        neigh = jnp.where(valid, v[order], -jnp.ones((M+1,), jnp.int32))
        ed    = jnp.where(valid[:, None], edge_attr[order], 100 * jnp.ones((M+1, edge_attr.shape[1])))
        return neigh[1:], ed[1:]

    A, edges = vmap(topM_per_node)(jnp.arange(N))
    return A, edges

edge_index, edge_vecs = topM_edges(edge_index, edge_attr, M=10)

def _smooth_cutoff_one(r, r_cs, r_c, eps=1e-8):
    r_safe = jnp.maximum(r, eps)
    inv_r  = 1.0 / r_safe
    u      = (r_safe - r_cs) / (r_c - r_cs)
    poly   = u**3 * (-6.0 * u**2 + 15.0 * u - 10.0)
    s_mid  = inv_r * (poly + 1.0)
    return jnp.where(r_safe < r_cs, inv_r, jnp.where(r_safe < r_c, s_mid, 0.0))

class EnvironmentMatrix(nnx.Module):
    """
    Compute environment matrix R^i (eq.4 of DP-Compress) for a single atom.

    inputs:
        edge_vecs : (N_cut, 4) the array of [r, x, y, z]
    """
    def __init__(self, r_cs, r_c, s_mean, s_std, corrds_std):
        self.r_cs = r_cs
        self.r_c  = r_c
        self.s_mean = s_mean
        self.s_std  = s_std
        self.corrds_std = corrds_std    
    
    def __call__(self, edge_vecs):
        sij = _smooth_cutoff_one(edge_vecs[:,0], self.r_cs, self.r_c)
        r0 = (sij - self.s_mean) / self.s_std
        rvec = edge_vecs[:,1:] * sij[:,None] / self.corrds_std

        return jnp.concatenate([r0[:,None], rvec], axis=-1) # (Ncut, 4)

class EmbeddingNet(nnx.Module):
    """
    Compute embedding matrix G^i for a single atom.

    inputs:
        shat: (N_cut, ) \hat{s}(r_{ij}), the first row of the env matrix.
    """
    def __init__(self, hidden: Sequence[int], M: int, *, rngs: nnx.Rngs):
        layers = []
        in_dim = 1
        for h in hidden:
            layers.append(nnx.Linear(in_dim, h, rngs=rngs))
            in_dim = h
        self.layers = layers
        self.out = nnx.Linear(in_dim, M, rngs=rngs)

    def __call__(self, shat: jnp.ndarray) -> jnp.ndarray:
        shat = shat[..., None]                    # (Ncut, 1)
        for lyr in self.layers:
            shat = jax.nn.relu(lyr(shat))
        return self.out(shat)                     # (Ncut, M)

class DescriptorNet(nnx.Module):
    """
    Compute descriptor matrix D^i for a single atom.
    """
    def __init__(self, 
                 r_cs, r_c, s_mean, s_std, corrds_std,
                 hidden: Sequence[int], M: int, Mp: int, rngs: nnx.Rngs):
        self.r_cs, self.r_c, self.s_mean, self.s_std, self.corrds_std = r_cs, r_c, s_mean, s_std, corrds_std
        self.hidden, self.M = hidden, M
        self.Mp = Mp
        self.env = EnvironmentMatrix(self.r_cs, self.r_c, self.s_mean, self.s_std, self.corrds_std)
        self.embed = EmbeddingNet(self.hidden, self.M, rngs=rngs)

    def __call__(self, edge_vecs: jnp.ndarray) -> jnp.ndarray:
        env = self.env(edge_vecs) # (Ncut, 4)
        Gi = self.embed(env[:, 0]) # (Ncut, M)
        Si = Gi.T @ env # (M, 4)
        Si_small = Si[:self.Mp] # (Mp, 4)
        Di = Si @ Si_small.T # (M, Mp)

        return Di

class FittingNet(nnx.Module):
    """
    Predict atomic energy E^i from descriptor D^i for a single atom.
    """
    def __init__(self, hidden: Sequence[int], M: int, Mp: int, *, rngs: nnx.Rngs):
        self.M, self.Mp = M, Mp
        self.hidden = hidden
        in_dim = M * Mp
        layers = []
        for h in hidden:
            layers.append(nnx.Linear(in_dim, h, rngs=rngs))
            in_dim = h
        self.layers = layers
        self.out = nnx.Linear(in_dim, 1, rngs=rngs)

    def __call__(self, Di: jnp.ndarray) -> jnp.ndarray:
        Di = Di.reshape(-1)                # flatten (M*Mp,)
        for lyr in self.layers:
            Di = jax.nn.relu(lyr(Di))
        return self.out(Di)                # (1,)

class EnergyNet(nnx.Module):
    def __init__(self, desc: DescriptorNet, fit: FittingNet):
        self.desc, self.fit = desc, fit

    def _one(self, edge_vecs):                  # edge_vecs: (N_cut, 4)
        return self.fit(self.desc(edge_vecs))   # (1,)

    def __call__(self, edge_vecs_batch):        # (N, N_cut, 4)
        return jax.vmap(self._one, in_axes=0, out_axes=0)(edge_vecs_batch)  # (N, 1)

r_cs = 2.0
r_c = 6.0
hidden_embed = [24, 48, 96]
hidden_fit   = [96, 96, 96]
s_mean = 0.0
s_std  = 1.0
corrds_std = 1.0
M = 16
Mp = 8
rngs = nnx.Rngs(0)
desc = DescriptorNet(r_cs, r_c, s_mean, s_std, corrds_std, hidden_embed, M, Mp, rngs=rngs)
fit  = FittingNet(hidden_fit, M, Mp, rngs=rngs)
net  = EnergyNet(desc, fit)
E = net(edge_vecs)   # edge_vecs.shape == (N, N_cut, 4) -> E.shape == (N, 1)
breakpoint()