from typing import Sequence

import torch
from torch_geometric.data import Data, InMemoryDataset

import jax
import optax
from jax import vmap
import jax.numpy as jnp
import flax.nnx as nnx
from flax.linen.initializers import variance_scaling, zeros as zeros_init
HE_INIT = variance_scaling(2.0, "fan_in", "truncated_normal")

from deepx.data.from_dft import AbInitDataSetFeature
from deepx.data.graphset import GraphSetInMemory, GraphSetOnDisk
from deepx.data.dataloader import get_loader_training

torch.set_default_dtype(torch.float32)

dft_data_feature = AbInitDataSetFeature("./ethanol-transfer/dft")
graph = GraphSetInMemory(
    graph_pt_path="./ethanol-transfer/dft/MD17.train-HE.train-IP.pt",
    dataset_name="MD17",
    graph_type="train-IP",
    graph_save_dir="./ethanol-transfer/graph",
    parallel_num=8,
    dft_data_feature=dft_data_feature.get_features(),
    forcibly_rebuild=True,
)
print(graph.info)
print("********************************")
random_stream = jax.random.PRNGKey(0)
train_loader, val_loader, test_loader = get_loader_training(
    graph, train_size=800, val_size=100, test_size=100, batch_size=32, graph_key=random_stream
)
batch = next(iter(train_loader))

atomic_numbers = jnp.array(batch.x)
edge_index = jnp.array(batch.edge_index)
edge_attr = jnp.array(batch.edge_attr[:,1:])  # (num_edges, 4) -> (num_edges, 3)

def topM_edges(edge_index, edge_attr, Ncut):
    """
    Get the top Ncut edges for each node based on their attributes.
    The input edge_attr is expected to be the edge vectors (dx, dy, dz). 
    We discard the original edge lengths for auto-differentiation.
    """
    u, v = edge_index
    N = int(jnp.max(edge_index)) + 1
    lens = jnp.linalg.norm(edge_attr, axis=1)

    def topM_per_node(i):
        mask  = u == i
        d     = jnp.where(mask, lens, jnp.inf)
        order = jnp.argsort(d)[:Ncut+1]
        valid = jnp.isfinite(d[order])
        neigh = jnp.where(valid, v[order], -jnp.ones((Ncut+1,), jnp.int32))
        ed    = jnp.where(valid[:, None], edge_attr[order], 100 * jnp.ones((Ncut+1, edge_attr.shape[1])))
        return neigh[1:], ed[1:]

    A, edges = vmap(topM_per_node)(jnp.arange(N))
    return A, edges

def prepare_batch(batch, Ncut=10):
    edge_index_raw = jnp.array(batch.edge_index)           # (2, E)
    edge_attr_raw  = jnp.array(batch.edge_attr[:, 1:])     # (E, 3) dx,dy,dz
    neigh_idx, edge_vecs = topM_edges(edge_index_raw, edge_attr_raw, Ncut)  # (N,Ncut), (N,Ncut,3)
    E_true_struct = jnp.asarray(batch.energy)              # (S,)
    F_true = jnp.asarray(batch.label)                      # (N,3)
    atom2struct = jnp.asarray(batch.batch)                 # (N,)
    return neigh_idx, edge_vecs, atom2struct, E_true_struct, F_true

edge_index, edge_vecs = topM_edges(edge_index, edge_attr, Ncut=10) # (N, Ncut), (N, Ncut, 3). These are the inputs to the model.

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
        edge_vecs : (N_cut, 3) the array of [x, y, z]
    """
    def __init__(self, r_cs, r_c, s_mean, s_std, corrds_std):
        self.r_cs = r_cs
        self.r_c  = r_c
        self.s_mean = s_mean
        self.s_std  = s_std
        self.corrds_std = corrds_std    
    
    def __call__(self, edge_vecs):
        lengths = jnp.linalg.norm(edge_vecs, axis=1)  # (Ncut, )
        sij = _smooth_cutoff_one(lengths, self.r_cs, self.r_c)
        r0 = (sij - self.s_mean) / self.s_std
        rvec = edge_vecs * sij[:,None] / self.corrds_std

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
            layers.append(nnx.Linear(in_dim, h, rngs=rngs,
                                     kernel_init=HE_INIT, bias_init=zeros_init))
            in_dim = h
        self.layers = layers
        self.out = nnx.Linear(in_dim, M, rngs=rngs,
                              kernel_init=HE_INIT, bias_init=zeros_init)

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
            layers.append(nnx.Linear(in_dim, h, rngs=rngs,
                                     kernel_init=HE_INIT, bias_init=zeros_init))
            in_dim = h
        self.layers = layers
        self.out = nnx.Linear(in_dim, 1, rngs=rngs,
                              kernel_init=HE_INIT, bias_init=zeros_init)

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

pe, pf = 1.0, 1.0
E_true_struct = jnp.asarray(batch.energy)   # (S,)
F_true = jnp.asarray(batch.label)           # (N,3)
atom2struct = jnp.asarray(batch.batch)      # (N,)
N, S = edge_vecs.shape[0], E_true_struct.shape[0]
neigh_idx = edge_index                      # (N, Ncut), padded with -1

optimizer = nnx.Optimizer(net, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, edge_vecs, neigh_idx, atom2struct, E_true_struct, F_true, pe, pf):
    def loss_fn(model):
        Ei = jnp.squeeze(model(edge_vecs), -1)                # (N,)
        S = E_true_struct.shape[0]
        N_per = jnp.zeros(S).at[atom2struct].add(1.0)
        Etot_pred = jnp.zeros(S).at[atom2struct].add(Ei)
        eps_pred, eps_true = Etot_pred / N_per, E_true_struct / N_per

        # forces
        def ei_fn(ev): return model.fit(model.desc(ev)).squeeze()
        dE_dedge = jax.vmap(jax.grad(ei_fn))(edge_vecs)       # (N,Ncut,3)
        own = dE_dedge.sum(axis=1)
        mask = (neigh_idx >= 0)[..., None]
        idx  = jnp.clip(neigh_idx, 0, edge_vecs.shape[0]-1).reshape(-1)
        contrib = (-dE_dedge * mask).reshape(-1, 3)
        others = jnp.zeros_like(own).at[idx].add(contrib)
        F_pred = own + others

        loss_e = pe * jnp.mean((eps_pred - eps_true) ** 2)
        loss_f = pf / (3.0 * edge_vecs.shape[0]) * jnp.sum((F_pred - F_true) ** 2)
        jax.debug.print("F_pred[0]={}, F_true[0]={}", F_pred[0], F_true[0])
        jax.debug.print("Etot_pred[0]={}, Etot_true[0]={}", Etot_pred[0], E_true_struct[0])
        return loss_e + loss_f

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

@nnx.jit
def eval_step(model, edge_vecs, neigh_idx, atom2struct, E_true_struct, F_true, pe, pf):
    def loss_fn(model):
        Ei = jnp.squeeze(model(edge_vecs), -1)                # (N,)
        S = E_true_struct.shape[0]
        N_per = jnp.zeros(S).at[atom2struct].add(1.0)
        Etot_pred = jnp.zeros(S).at[atom2struct].add(Ei)
        eps_pred, eps_true = Etot_pred / N_per, E_true_struct / N_per

        # forces
        def ei_fn(ev): return model.fit(model.desc(ev)).squeeze()
        dE_dedge = jax.vmap(jax.grad(ei_fn))(edge_vecs)       # (N,Ncut,3)
        own = dE_dedge.sum(axis=1)
        mask = (neigh_idx >= 0)[..., None]
        idx  = jnp.clip(neigh_idx, 0, edge_vecs.shape[0]-1).reshape(-1)
        contrib = (-dE_dedge * mask).reshape(-1, 3)
        others = jnp.zeros_like(own).at[idx].add(contrib)
        F_pred = own + others

        loss_e = pe * jnp.mean((eps_pred - eps_true) ** 2)
        loss_f = pf / (3.0 * edge_vecs.shape[0]) * jnp.sum((F_pred - F_true) ** 2)
        return loss_e + loss_f
    return loss_fn(model)

num_epochs = 1000
Ncut = 21

for epoch in range(1, num_epochs + 1):
    ########### Train ########### 
    net.train()
    train_loss_sum, train_batches = 0.0, 0
    for b in train_loader:
        neigh_idx_b, edge_vecs_b, atom2struct_b, E_true_b, F_true_b = prepare_batch(b, Ncut)
        loss = train_step(net, optimizer, edge_vecs_b, neigh_idx_b, atom2struct_b, E_true_b, F_true_b, pe, pf)
        train_loss_sum += float(loss); train_batches += 1
    train_loss = train_loss_sum / max(1, train_batches)

    ########### Val ###########
    net.eval()
    val_loss_sum, val_batches = 0.0, 0
    for b in val_loader:
        neigh_idx_b, edge_vecs_b, atom2struct_b, E_true_b, F_true_b = prepare_batch(b, Ncut)
        vloss = eval_step(net, edge_vecs_b, neigh_idx_b, atom2struct_b, E_true_b, F_true_b, pe, pf)
        val_loss_sum += float(vloss); val_batches += 1
    val_loss = val_loss_sum / max(1, val_batches)

    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f}")

########### Test ###########
net.eval()
test_loss_sum, test_batches = 0.0, 0
for b in test_loader:
    neigh_idx_b, edge_vecs_b, atom2struct_b, E_true_b, F_true_b = prepare_batch(b, Ncut)
    tloss = eval_step(net, edge_vecs_b, neigh_idx_b, atom2struct_b, E_true_b, F_true_b, pe, pf)
    test_loss_sum += float(tloss); test_batches += 1
test_loss = test_loss_sum / max(1, test_batches)
print(f"[TEST] loss = {test_loss:.6f}")