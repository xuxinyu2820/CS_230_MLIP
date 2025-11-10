from typing import Sequence

import torch
from torch_geometric.data import Data, InMemoryDataset

import jax
from jax import vmap
import jax.numpy as jnp
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import flax.nnx as nnx
from flax.linen.initializers import variance_scaling, zeros as zeros_init
GLOROT_INIT = variance_scaling(1.0, "fan_avg", "truncated_normal", dtype=jnp.float64)
HE_INIT = variance_scaling(2.0, "fan_in", "truncated_normal", dtype=jnp.float64)

import optax

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
    graph, train_size=800, val_size=100, test_size=100, batch_size=4, graph_key=random_stream
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
        ed    = jnp.where(valid[:, None], edge_attr[order], 100 * jnp.ones((Ncut+1, edge_attr.shape[1]), dtype=edge_attr.dtype))
        return neigh[1:], ed[1:]

    A, edges = vmap(topM_per_node)(jnp.arange(N))
    return A, edges

def prepare_batch(batch, Ncut=10):
    edge_index_raw = jnp.array(batch.edge_index)                 # (2, E)
    edge_attr_raw  = jnp.array(batch.edge_attr[:, 1:], dtype=jnp.float64)  # (E,3)
    neigh_idx, edge_vecs = topM_edges(edge_index_raw, edge_attr_raw, Ncut)  # (N,Ncut),(N,Ncut,3)

    Z = jnp.asarray(batch.x, dtype=jnp.int32)                    # (N,)
    sort_key = jnp.where(neigh_idx >= 0, Z[neigh_idx], jnp.iinfo(jnp.int32).max)
    order = jnp.argsort(sort_key, axis=1)                        # (N,Ncut)
    neigh_idx = jnp.take_along_axis(neigh_idx, order, axis=1)
    edge_vecs  = jnp.take_along_axis(edge_vecs,  order[..., None], axis=1)

    E_true_struct = jnp.asarray(batch.energy, dtype=jnp.float64) # (S,)
    F_true        = jnp.asarray(batch.label,  dtype=jnp.float64) # (N,3)
    atom2struct   = jnp.asarray(batch.batch)                     # (N,)
    return neigh_idx, edge_vecs, atom2struct, E_true_struct, F_true

edge_index, edge_vecs = topM_edges(edge_index, edge_attr, Ncut=10) # (N, Ncut), (N, Ncut, 3). These are the inputs to the model.

def _smooth_cutoff_one(r, r_cs, r_c, eps=1e-8):
    r_safe = jnp.maximum(r, eps)
    inv_r  = 1.0 / r_safe
    u      = (r_safe - r_cs) / (r_c - r_cs)
    poly   = u**3 * (-6.0 * u**2 + 15.0 * u - 10.0)
    s_mid  = inv_r * (poly + 1.0)
    return jnp.where(r_safe < r_cs, inv_r, jnp.where(r_safe < r_c, s_mid, 0.0))

def compute_env_stats(train_loader, r_cs, r_c, eps=1e-6):
    sum_s = 0.0; sumsq_s = 0.0; cnt_s = 0
    sum_xyz = 0.0; sumsq_xyz = 0.0; cnt_xyz = 0
    for b in train_loader:
        vec = jnp.array(b.edge_attr[:, 1:], dtype=jnp.float64)     # (E,3): dx,dy,dz
        r = jnp.linalg.norm(vec, axis=1)                           # (E,)
        valid = (r > eps) & (r < r_c)                              # (E,) 
        if not bool(valid.any()):
            continue
        r = r[valid]; vec = vec[valid]

        s = _smooth_cutoff_one(r, r_cs, r_c)                       # s(r_ij)
        # 式(2)：s(r) 的均值与标准差
        sum_s += float(s.sum())
        sumsq_s += float(jnp.sum(s * s))
        cnt_s += int(s.size)

        # 式(3)：把 {s x, s y, s z} 合成一个大样本求整体标准差
        svec = (s[:, None] * vec).reshape(-1)                      # (3*E_valid,)
        sum_xyz += float(svec.sum())
        sumsq_xyz += float(jnp.dot(svec, svec))
        cnt_xyz += int(svec.size)

    s_mean = sum_s / max(1, cnt_s)
    var_s = max(sumsq_s / max(1, cnt_s) - s_mean * s_mean, 1e-12)
    s_std = var_s ** 0.5

    mu_xyz = sum_xyz / max(1, cnt_xyz)
    var_xyz = max(sumsq_xyz / max(1, cnt_xyz) - mu_xyz * mu_xyz, 1e-12)
    corrds_std = var_xyz ** 0.5
    return float(s_mean), float(s_std), float(corrds_std)

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
                                     kernel_init=HE_INIT, bias_init=zeros_init, param_dtype=jnp.float64))
            in_dim = h
        self.layers = layers
        self.out = nnx.Linear(in_dim, M, rngs=rngs,
                              kernel_init=HE_INIT, bias_init=zeros_init, param_dtype=jnp.float64)

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
                                     kernel_init=HE_INIT, bias_init=zeros_init, param_dtype=jnp.float64))
            in_dim = h
        self.layers = layers
        self.out = nnx.Linear(in_dim, 1, rngs=rngs,
                              kernel_init=HE_INIT, bias_init=zeros_init, param_dtype=jnp.float64)

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
s_mean, s_std, corrds_std = compute_env_stats(train_loader, r_cs, r_c)
print(f"[stats] s_mean={s_mean:.6f}, s_std={s_std:.6f}, corrds_std={corrds_std:.6f}")
breakpoint()
M = 100
Mp = 10
rngs = nnx.Rngs(0)
desc = DescriptorNet(r_cs, r_c, s_mean, s_std, corrds_std, hidden_embed, M, Mp, rngs=rngs)
fit  = FittingNet(hidden_fit, M, Mp, rngs=rngs)
net  = EnergyNet(desc, fit)
E = net(edge_vecs)   # edge_vecs.shape == (N, N_cut, 4) -> E.shape == (N, 1)


r0 = 1e-3
decay_rate = 0.95
decay_step = 40000
lr_schedule = optax.exponential_decay(
    init_value=r0, transition_steps=decay_step, decay_rate=decay_rate, staircase=True
)
optimizer = nnx.Optimizer(net, optax.adam(lr_schedule), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, edge_vecs, neigh_idx, atom2struct, E_true_struct, F_true, pe, pf):
    def loss_fn(model):
        Ei = jnp.squeeze(model(edge_vecs), -1)                # (N,)
        S = E_true_struct.shape[0]
        N_per = jnp.zeros(S, dtype=jnp.float64).at[atom2struct].add(1.0)
        Etot_pred = jnp.zeros(S, dtype=jnp.float64).at[atom2struct].add(Ei)
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

num_epochs = 2000
Ncut = 21
pe_start, pe_limit = 1.0, 400.0
pf_start, pf_limit = 1000.0, 1.0

global_step = 0
for epoch in range(1, num_epochs + 1):
    net.train()
    train_loss_sum, train_batches = 0.0, 0
    for b in train_loader:
        neigh_idx_b, edge_vecs_b, atom2struct_b, E_true_b, F_true_b = prepare_batch(b, Ncut)

        rt = lr_schedule(global_step)
        alpha = rt / r0
        pe_cur = pe_limit * (1 - alpha) + pe_start * alpha
        pf_cur = pf_limit * (1 - alpha) + pf_start * alpha

        loss = train_step(net, optimizer, edge_vecs_b, neigh_idx_b,
                          atom2struct_b, E_true_b, F_true_b,
                          float(pe_cur), float(pf_cur))
        train_loss_sum += float(loss); train_batches += 1
        global_step += 1

    train_loss = train_loss_sum / max(1, train_batches)

    net.eval()
    val_loss_sum, val_batches = 0.0, 0
    for b in val_loader:
        neigh_idx_b, edge_vecs_b, atom2struct_b, E_true_b, F_true_b = prepare_batch(b, Ncut)
        vloss = eval_step(net, edge_vecs_b, neigh_idx_b, atom2struct_b, E_true_b, F_true_b,
                          float(pe_cur), float(pf_cur))
        val_loss_sum += float(vloss); val_batches += 1
    val_loss = val_loss_sum / max(1, val_batches)

    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch:04d} | lr={float(rt):.3e} | pe={float(pe_cur):.2f} pf={float(pf_cur):.2f} | "
              f"train {train_loss:.6f} | val {val_loss:.6f}")

########### Test ###########
net.eval()
test_loss_sum, test_batches = 0.0, 0
for b in test_loader:
    neigh_idx_b, edge_vecs_b, atom2struct_b, E_true_b, F_true_b = prepare_batch(b, Ncut)
    tloss = eval_step(net, edge_vecs_b, neigh_idx_b, atom2struct_b, E_true_b, F_true_b, pe, pf)
    test_loss_sum += float(tloss); test_batches += 1
test_loss = test_loss_sum / max(1, test_batches)
print(f"[TEST] loss = {test_loss:.6f}")