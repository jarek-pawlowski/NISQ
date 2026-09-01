#!/usr/bin/env python3
"""
Utilities for the N=6 learned SW-like witness experiment.

The public script `swssb_n6_continuous_matching_train.py` is intentionally
small. Physics, data generation, matching, ML, diagnostics, and the final
trajectory demo live here.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

N = 6
DIM = 2**N
N_EXC_INITIAL = 3
EPS_C = 0.02
EPS_R2 = 0.05
DEFAULT_SEED = 20260901


# ---------------------------------------------------------------------------
# Operators and charge sector
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SP2 = 0.5 * (X2 + 1j * Y2)
SM2 = 0.5 * (X2 - 1j * Y2)
PAULI_1Q = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def kron_all_dense(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def op_on_site_dense(op, site):
    return kron_all_dense([op if i == site else I2 for i in range(N)])


def two_site_dense(op_a, site_a, op_b, site_b):
    ops = []
    for i in range(N):
        if i == site_a:
            ops.append(op_a)
        elif i == site_b:
            ops.append(op_b)
        else:
            ops.append(I2)
    return kron_all_dense(ops)


X_SITE = [op_on_site_dense(X2, i) for i in range(N)]
Y_SITE = [op_on_site_dense(Y2, i) for i in range(N)]
Z_SITE = [op_on_site_dense(Z2, i) for i in range(N)]
SP_SITE = [op_on_site_dense(SP2, i) for i in range(N)]
SM_SITE = [op_on_site_dense(SM2, i) for i in range(N)]

ID = np.eye(DIM, dtype=np.complex128)
ID_S = sparse.identity(DIM, dtype=np.complex128, format="csr")


def index_bits(index):
    return tuple((index >> (N - 1 - k)) & 1 for k in range(N))


BASIS_BITS = [index_bits(i) for i in range(DIM)]
N_EXC = np.array([sum(bits) for bits in BASIS_BITS], dtype=int)

Q_OP = np.diag(N_EXC.astype(np.float64)).astype(np.complex128)
Q2_OP = Q_OP @ Q_OP

P_INITIAL_SECTOR = np.zeros((DIM, DIM), dtype=np.complex128)
for idx in np.where(N_EXC == N_EXC_INITIAL)[0]:
    P_INITIAL_SECTOR[idx, idx] = 1.0

A_1N = SP_SITE[0] @ SM_SITE[-1]


# ---------------------------------------------------------------------------
# Initial states and Hamiltonian
# ---------------------------------------------------------------------------

def random_fixed_sector_rho(rng):
    basis = np.where(N_EXC == N_EXC_INITIAL)[0]
    coeff = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    coeff /= np.linalg.norm(coeff)
    psi = np.zeros(DIM, dtype=np.complex128)
    psi[basis] = coeff
    return np.outer(psi, psi.conj())


def basis_state_rho(bitstring="101010"):
    """Return |bitstring><bitstring|; the default has exactly three excitations."""
    if len(bitstring) != N or any(c not in "01" for c in bitstring):
        raise ValueError(f"bitstring must contain exactly {N} bits")
    if sum(int(c) for c in bitstring) != N_EXC_INITIAL:
        raise ValueError(
            f"demo initial state must lie in Q={N_EXC_INITIAL}; got {bitstring}"
        )
    idx = int(bitstring, 2)
    psi = np.zeros(DIM, dtype=np.complex128)
    psi[idx] = 1.0
    return np.outer(psi, psi.conj())


def build_hamiltonian(Jxy, Jz, h):
    H = np.zeros((DIM, DIM), dtype=np.complex128)
    for i in range(N - 1):
        H += Jxy[i] * (X_SITE[i] @ X_SITE[i + 1] + Y_SITE[i] @ Y_SITE[i + 1])
        H += Jz[i] * (Z_SITE[i] @ Z_SITE[i + 1])
    for i in range(N):
        H += h[i] * Z_SITE[i]
    return H


# ---------------------------------------------------------------------------
# Lindbladian
# ---------------------------------------------------------------------------

def csr(a):
    return sparse.csr_matrix(a)


def dissipator_superoperator(L_dense):
    L = csr(L_dense)
    LdagL = L.getH() @ L
    return (
        sparse.kron(L.conjugate(), L, format="csr")
        - 0.5 * sparse.kron(ID_S, LdagL, format="csr")
        - 0.5 * sparse.kron(LdagL.transpose(), ID_S, format="csr")
    )


D_DEPH = sparse.csr_matrix((DIM * DIM, DIM * DIM), dtype=np.complex128)
D_LOSS = sparse.csr_matrix((DIM * DIM, DIM * DIM), dtype=np.complex128)
D_GAIN = sparse.csr_matrix((DIM * DIM, DIM * DIM), dtype=np.complex128)
D_HOP = sparse.csr_matrix((DIM * DIM, DIM * DIM), dtype=np.complex128)

for i in range(N):
    D_DEPH += dissipator_superoperator(Z_SITE[i])
    D_LOSS += dissipator_superoperator(SM_SITE[i])
    D_GAIN += dissipator_superoperator(SP_SITE[i])

for i in range(N - 1):
    D_HOP += dissipator_superoperator(SP_SITE[i + 1] @ SM_SITE[i])
    D_HOP += dissipator_superoperator(SP_SITE[i] @ SM_SITE[i + 1])


def liouvillian(H, gamma_phi, gamma_hop, gamma_loss, gamma_gain):
    Hs = csr(H)
    coherent = -1j * (
        sparse.kron(ID_S, Hs, format="csr")
        - sparse.kron(Hs.transpose(), ID_S, format="csr")
    )
    return (
        coherent
        + gamma_phi * D_DEPH
        + gamma_hop * D_HOP
        + gamma_loss * D_LOSS
        + gamma_gain * D_GAIN
    )


def rho_to_vec(rho):
    return rho.T.reshape(-1)


def vec_to_rho(vec):
    rho = vec.reshape(DIM, DIM).T.copy()
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-14:
        rho /= tr
    return rho


# ---------------------------------------------------------------------------
# SW diagnostics and Pauli measurements
# ---------------------------------------------------------------------------

def linear_correlator(rho):
    return float(abs(np.trace(rho @ A_1N)))


def renyi2_correlator(rho):
    numerator = np.real(np.trace(rho @ A_1N @ rho @ A_1N.conj().T))
    purity = np.real(np.trace(rho @ rho))
    return float(numerator / (purity + 1e-12))


def charge_diagnostics(rho):
    q = float(np.real(np.trace(rho @ Q_OP)))
    q2 = float(np.real(np.trace(rho @ Q2_OP)))
    var_q = q2 - q * q
    p_sector = float(np.real(np.trace(rho @ P_INITIAL_SECTOR)))
    return q, var_q, 1.0 - p_sector


def generate_local_pauli_labels():
    labels = []
    for i in range(N):
        for p in "XYZ":
            chars = ["I"] * N
            chars[i] = p
            labels.append("".join(chars))
    for i, j in itertools.combinations(range(N), 2):
        for p in "XYZ":
            for q in "XYZ":
                chars = ["I"] * N
                chars[i] = p
                chars[j] = q
                labels.append("".join(chars))
    return labels


PAULI_LABELS = generate_local_pauli_labels()


def pauli_matrix(label):
    return kron_all_dense([PAULI_1Q[s] for s in label])


def sparse_trace_recipe(op):
    rows, cols = np.nonzero(np.abs(op) > 1e-14)
    return rows.astype(np.int32), cols.astype(np.int32), op[rows, cols]


PAULI_RECIPES = [sparse_trace_recipe(pauli_matrix(label)) for label in PAULI_LABELS]


def local_pauli_features(rho):
    out = np.empty(len(PAULI_RECIPES), dtype=np.float32)
    for k, (rows, cols, vals) in enumerate(PAULI_RECIPES):
        out[k] = np.real(np.sum(vals * rho[cols, rows]))
    return out


# ---------------------------------------------------------------------------
# Targeted trajectory generator
# ---------------------------------------------------------------------------

def make_targeted_time_grid(n_times, t_max, late_time_fraction, late_time_start):
    if n_times < 2:
        return np.array([0.0], dtype=float)
    if not 0.0 <= late_time_fraction <= 1.0:
        raise ValueError("late_time_fraction must be in [0,1]")
    if not 0.0 < late_time_start < t_max:
        raise ValueError("late_time_start must satisfy 0 < start < t_max")

    n_nonzero = n_times - 1
    n_late = min(max(int(round(late_time_fraction * n_nonzero)), 1), n_nonzero)
    n_early = n_nonzero - n_late

    times = [0.0]
    if n_early:
        times.extend(
            np.linspace(0.0, late_time_start, n_early + 2, endpoint=True)[1:-1].tolist()
        )
    times.extend(
        np.linspace(late_time_start, t_max, n_late + 1, endpoint=True)[1:].tolist()
    )
    return np.asarray(times, dtype=float)


def evolve_at_times(L, rho0, times):
    """Robust arbitrary-time evolution; used for final trajectory demos."""
    v0 = rho_to_vec(rho0)
    states = []
    for t in times:
        if abs(float(t)) < 1e-14:
            v = v0
        else:
            v = expm_multiply(L * float(t), v0)
        states.append(vec_to_rho(v))
    return states


def evolve_at_targeted_times(L, vec0, time_grid, late_time_start):
    """
    Efficient two-segment evolution for the targeted training time grid.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    if len(time_grid) == 1:
        return np.asarray([vec0])

    early_times = time_grid[(time_grid > 0.0) & (time_grid < late_time_start - 1e-12)]
    late_times = time_grid[time_grid >= late_time_start - 1e-12]
    results = {0.0: vec0}

    if len(early_times):
        early_full = np.concatenate([[0.0], early_times])
        if len(early_full) == 2:
            results[float(early_full[-1])] = expm_multiply(
                L * float(early_full[-1]), vec0
            )
        else:
            early_vecs = expm_multiply(
                L, vec0, start=0.0, stop=float(early_full[-1]),
                num=len(early_full), endpoint=True
            )
            for t, v in zip(early_full, early_vecs):
                results[float(t)] = v

    v_late0 = expm_multiply(L * late_time_start, vec0)
    results[float(late_time_start)] = v_late0

    late_after_start = late_times[late_times > late_time_start + 1e-12]
    if len(late_after_start):
        rel_stop = float(late_after_start[-1] - late_time_start)
        if len(late_after_start) == 1:
            late_vecs = [expm_multiply(L * rel_stop, v_late0)]
        else:
            late_vecs = expm_multiply(
                L, v_late0, start=0.0, stop=rel_stop,
                num=len(late_after_start) + 1, endpoint=True
            )[1:]
        for t, v in zip(late_after_start, late_vecs):
            results[float(t)] = v

    ordered = []
    for t in time_grid:
        key = min(results.keys(), key=lambda x: abs(x - float(t)))
        if abs(key - float(t)) > 1e-8:
            ordered.append(expm_multiply(L * float(t), vec0))
        else:
            ordered.append(results[key])
    return np.asarray(ordered)


def sample_targeted_parameters(rng, regime_a_fraction=0.5):
    """Sample one trajectory from the same physical family used for training."""
    Jxy = rng.uniform(0.6, 1.4, size=N - 1)
    Jz = rng.uniform(0.4, 1.2, size=N - 1)
    h = rng.uniform(-0.5, 0.5, size=N)
    gamma_phi = float(rng.uniform(0.02, 0.35))
    gamma_hop = float(rng.uniform(0.00, 0.30))

    use_a = bool(rng.random() < regime_a_fraction)
    if use_a:
        gamma_gain = float(rng.uniform(0.00, 0.025))
        gamma_loss = float(rng.uniform(0.15, 0.35))
        regime = 0
    else:
        gamma_gain = float(rng.uniform(0.025, 0.18))
        gamma_loss = float(rng.uniform(0.02, 0.25))
        regime = 1

    return dict(
        Jxy=Jxy, Jz=Jz, h=h,
        gamma_phi=gamma_phi, gamma_hop=gamma_hop,
        gamma_loss=gamma_loss, gamma_gain=gamma_gain,
        regime=regime,
    )


def generate_raw_dataset(
    n_trajectories,
    n_times,
    t_max,
    seed,
    regime_a_fraction=0.50,
    late_time_fraction=0.70,
    late_time_start=5.0,
):
    rng = np.random.default_rng(seed)
    n_total = n_trajectories * n_times

    X = np.empty((n_total, len(PAULI_LABELS)), dtype=np.float32)
    scalar_keys = (
        "linear", "r2", "q_mean", "var_q", "p_out", "time",
        "gamma_phi", "gamma_hop", "gamma_loss", "gamma_gain"
    )
    scalars = {k: np.empty(n_total, dtype=np.float32) for k in scalar_keys}
    traj_id = np.empty(n_total, dtype=np.int32)
    regime_all = np.empty(n_total, dtype=np.int8)

    time_grid = make_targeted_time_grid(
        n_times, t_max, late_time_fraction, late_time_start
    )
    print("targeted time grid =", np.array2string(time_grid, precision=3))

    write = 0
    t0 = time.time()

    for tr in range(n_trajectories):
        rho0 = random_fixed_sector_rho(rng)
        pars = sample_targeted_parameters(rng, regime_a_fraction)
        H = build_hamiltonian(pars["Jxy"], pars["Jz"], pars["h"])
        L = liouvillian(
            H, pars["gamma_phi"], pars["gamma_hop"],
            pars["gamma_loss"], pars["gamma_gain"]
        )
        vecs = evolve_at_targeted_times(L, rho_to_vec(rho0), time_grid, late_time_start)

        for it, vec in enumerate(vecs):
            rho = vec_to_rho(vec)
            X[write] = local_pauli_features(rho)
            scalars["linear"][write] = linear_correlator(rho)
            scalars["r2"][write] = renyi2_correlator(rho)
            (
                scalars["q_mean"][write],
                scalars["var_q"][write],
                scalars["p_out"][write],
            ) = charge_diagnostics(rho)
            scalars["time"][write] = time_grid[it]
            for name in ("gamma_phi", "gamma_hop", "gamma_loss", "gamma_gain"):
                scalars[name][write] = pars[name]
            traj_id[write] = tr
            regime_all[write] = pars["regime"]
            write += 1

        if (
            tr == 0
            or (tr + 1) % max(1, n_trajectories // 20) == 0
            or tr + 1 == n_trajectories
        ):
            elapsed = time.time() - t0
            rate = (tr + 1) / max(elapsed, 1e-12)
            eta = (n_trajectories - tr - 1) / max(rate, 1e-12)
            print(
                f"trajectory {tr + 1:5d}/{n_trajectories} | "
                f"states={write:7d}/{n_total} | "
                f"elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min"
            )

    r2 = scalars["r2"]
    return {
        "X": X,
        **scalars,
        "y": (r2 > EPS_R2).astype(np.float32),
        "traj_id": traj_id,
        "regime": regime_all,
    }


# ---------------------------------------------------------------------------
# Conditioning and continuous charge matching
# ---------------------------------------------------------------------------

def subset_dict(data, idx):
    return {key: value[idx] for key, value in data.items()}


def condition_on_low_clin(data):
    idx = np.where(data["linear"] < EPS_C)[0]
    out = subset_dict(data, idx)
    out["y"] = (out["r2"] > EPS_R2).astype(np.float32)
    out["original_index"] = idx
    return out


def split_trajectories_raw(data, seed, train_frac=0.70, val_frac=0.15):
    rng = np.random.default_rng(seed)
    unique_traj = np.unique(data["traj_id"])
    rng.shuffle(unique_traj)

    n = len(unique_traj)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    groups = (
        set(unique_traj[:n_train].tolist()),
        set(unique_traj[n_train:n_train+n_val].tolist()),
        set(unique_traj[n_train+n_val:].tolist()),
    )

    def take(trset):
        mask = np.array([int(tr) in trset for tr in data["traj_id"]])
        return subset_dict(data, np.where(mask)[0])

    return tuple(take(g) for g in groups)


def greedy_charge_match(
    data,
    eps_pout,
    eps_varq,
    eps_qmean,
    seed,
    k_neighbors=64,
):
    """
    One-to-one nearest-neighbour matching between y=0 and y=1 in
    (p_out, Var(Q), <Q>). Matched partners must come from different trajectories.
    """
    rng = np.random.default_rng(seed)
    y = data["y"].astype(int)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if len(i0) == 0 or len(i1) == 0:
        return None, {"pairs": 0, "states": 0}

    scale = np.array([eps_pout, eps_varq, eps_qmean], dtype=float)
    if np.any(scale <= 0):
        raise ValueError("all matching tolerances must be positive")

    def features(idx):
        return np.column_stack(
            [data["p_out"][idx], data["var_q"][idx], data["q_mean"][idx]]
        ) / scale

    if len(i0) <= len(i1):
        anchor_idx, pool_idx, anchor_label = i0.copy(), i1.copy(), 0
    else:
        anchor_idx, pool_idx, anchor_label = i1.copy(), i0.copy(), 1

    anchor_f, pool_f = features(anchor_idx), features(pool_idx)
    tree = cKDTree(pool_f)
    used_pool = np.zeros(len(pool_idx), dtype=bool)
    pairs = []

    k_query = min(max(1, int(k_neighbors)), len(pool_idx))
    for aa in rng.permutation(len(anchor_idx)):
        _, neigh = tree.query(anchor_f[aa], k=k_query)
        a_idx = int(anchor_idx[aa])
        chosen = None

        for pp in np.atleast_1d(neigh):
            pp = int(pp)
            if used_pool[pp]:
                continue
            b_idx = int(pool_idx[pp])
            if int(data["traj_id"][a_idx]) == int(data["traj_id"][b_idx]):
                continue

            dp = abs(float(data["p_out"][a_idx] - data["p_out"][b_idx]))
            dv = abs(float(data["var_q"][a_idx] - data["var_q"][b_idx]))
            dq = abs(float(data["q_mean"][a_idx] - data["q_mean"][b_idx]))
            if dp <= eps_pout and dv <= eps_varq and dq <= eps_qmean:
                chosen = pp
                break

        if chosen is None:
            continue
        used_pool[chosen] = True
        b_idx = int(pool_idx[chosen])
        pairs.append((a_idx, b_idx) if anchor_label == 0 else (b_idx, a_idx))

    if not pairs:
        return None, {"pairs": 0, "states": 0}

    pairs = np.asarray(pairs, dtype=np.int64)
    selected = pairs.reshape(-1)
    matched = {key: value[selected] for key, value in data.items()}
    matched["pair_id"] = np.repeat(np.arange(len(pairs), dtype=np.int64), 2)

    dp = np.abs(data["p_out"][pairs[:, 0]] - data["p_out"][pairs[:, 1]])
    dv = np.abs(data["var_q"][pairs[:, 0]] - data["var_q"][pairs[:, 1]])
    dq = np.abs(data["q_mean"][pairs[:, 0]] - data["q_mean"][pairs[:, 1]])

    info = {
        "pairs": int(len(pairs)),
        "states": int(2 * len(pairs)),
        "mean_abs_dpout": float(np.mean(dp)),
        "max_abs_dpout": float(np.max(dp)),
        "mean_abs_dvarq": float(np.mean(dv)),
        "max_abs_dvarq": float(np.max(dv)),
        "mean_abs_dqmean": float(np.mean(dq)),
        "max_abs_dqmean": float(np.max(dq)),
    }
    return matched, info


def merge_matched_splits(train, val, test):
    common_keys = [key for key in train if key in val and key in test]
    merged = {
        key: np.concatenate([train[key], val[key], test[key]], axis=0)
        for key in common_keys
    }
    n_train, n_val, n_test = len(train["y"]), len(val["y"]), len(test["y"])
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)
    return merged, train_idx, val_idx, test_idx


def prepare_matched_dataset(
    raw,
    eps_pout,
    eps_varq,
    eps_qmean,
    k_neighbors,
    seed,
):
    """
    High-level wrapper used by main:
      trajectory split -> C_lin conditioning -> matching in each split.
    """
    raw_train, raw_val, raw_test = split_trajectories_raw(raw, seed=seed + 2)
    conditioned = [
        condition_on_low_clin(x) for x in (raw_train, raw_val, raw_test)
    ]

    matched = []
    infos = []
    for j, cond in enumerate(conditioned):
        m, info = greedy_charge_match(
            cond,
            eps_pout=eps_pout,
            eps_varq=eps_varq,
            eps_qmean=eps_qmean,
            seed=seed + 11 + j,
            k_neighbors=k_neighbors,
        )
        if m is None:
            raise RuntimeError(
                "matching produced an empty split; increase trajectories "
                "or relax matching tolerances"
            )
        matched.append(m)
        infos.append(info)

    merged, train_idx, val_idx, test_idx = merge_matched_splits(*matched)

    n_conditioned = sum(len(x["y"]) for x in conditioned)
    n0 = sum(int(np.sum(x["y"] == 0)) for x in conditioned)
    n1 = sum(int(np.sum(x["y"] == 1)) for x in conditioned)

    info = {
        "n_conditioned": n_conditioned,
        "conditioned_class0": n0,
        "conditioned_class1": n1,
        "conditioned_positive_fraction": n1 / max(n_conditioned, 1),
        "train": infos[0],
        "val": infos[1],
        "test": infos[2],
    }
    return merged, train_idx, val_idx, test_idx, info


# ---------------------------------------------------------------------------
# Learned sparse witness
# ---------------------------------------------------------------------------

class HardTopKSelector(nn.Module):
    def __init__(self, n_features, K, temperature=1.0):
        super().__init__()
        self.K = int(K)
        self.temperature = float(temperature)
        self.logits = nn.Parameter(1e-3 * torch.randn(n_features))

    def soft_scores(self):
        return torch.sigmoid(self.logits / self.temperature)

    def hard_mask(self):
        scores = self.soft_scores()
        inds = torch.topk(scores, self.K).indices
        hard = torch.zeros_like(scores)
        hard[inds] = 1.0
        return hard

    def forward(self, x):
        soft, hard = self.soft_scores(), self.hard_mask()
        mask = hard + soft - soft.detach()
        return x * mask, hard


class SWWitness(nn.Module):
    def __init__(self, n_features, K):
        super().__init__()
        self.selector = HardTopKSelector(n_features, K)
        self.reader = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x_sel, hard = self.selector(x)
        return self.reader(x_sel).squeeze(-1), hard


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)


def orientation_free_auc(y, score):
    auc = safe_auc(y, score)
    return np.nan if not np.isfinite(auc) else max(auc, 1.0 - auc)


@torch.no_grad()
def evaluate_model(model, X, y, batch_size=4096):
    model.eval()
    probs = []
    for start in range(0, len(X), batch_size):
        logits, _ = model(X[start:start+batch_size])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    prob = np.concatenate(probs)
    yy = y.cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": safe_auc(yy, prob),
        "acc": accuracy_score(yy, pred),
        "prob": prob,
        "y": yy,
    }


def normalize_matched_features(data, train_idx):
    mean = data["X"][train_idx].mean(axis=0, keepdims=True)
    std = data["X"][train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X = torch.tensor((data["X"] - mean) / std, dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)
    return X, y, mean, std


def train_model(X_train, y_train, X_val, y_val, K, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    model = SWWitness(n_features=X_train.shape[1], K=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    history = {"loss": [], "val_auc": []}
    best_auc, best_state = -np.inf, None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        val = evaluate_model(model, X_val, y_val)
        mean_loss = float(np.mean(losses))
        history["loss"].append(mean_loss)
        history["val_auc"].append(val["auc"])

        if np.isfinite(val["auc"]) and val["auc"] > best_auc:
            best_auc = val["auc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d} | loss={mean_loss:.5f} | val AUC={val['auc']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_auc


def selected_observables(model):
    with torch.no_grad():
        hard = model.selector.hard_mask().cpu().numpy()
    idx = np.where(hard > 0.5)[0]
    return idx, [PAULI_LABELS[i] for i in idx]


# ---------------------------------------------------------------------------
# Diagnostics / output
# ---------------------------------------------------------------------------

def save_training_history(history, output_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(history["loss"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("training BCE")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_history.png", dpi=180)
    plt.close(fig)


def save_test_roc(y, prob, auc, output_dir):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "test_roc.png", dpi=180)
    plt.close(fig)


def save_charge_auc_plot(witness_auc, q_auc, var_auc, pout_auc, output_dir):
    names = ["witness", "<Q>", "Var(Q)", "p_out"]
    vals = [witness_auc, q_auc, var_auc, pout_auc]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(np.arange(len(names)), vals)
    ax.axhline(0.5, linestyle="--")
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "test_charge_auc.png", dpi=180)
    plt.close(fig)


def save_probability_plot(y, prob, output_dir):
    rng = np.random.default_rng(DEFAULT_SEED)
    x = y + rng.normal(0.0, 0.03, size=len(y))
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.scatter(x, prob, s=15, alpha=0.45)
    ax.axhline(0.5, linestyle="--")
    ax.set_xticks([0, 1], ["non-SW-like", "SW-like"])
    ax.set_ylabel(r"$P_{\rm SW}$")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "test_probability.png", dpi=180)
    plt.close(fig)


def save_standard_diagnostics(history, test, q_auc, var_auc, pout_auc, output_dir):
    save_training_history(history, output_dir)
    save_test_roc(test["y"], test["prob"], test["auc"], output_dir)
    save_charge_auc_plot(test["auc"], q_auc, var_auc, pout_auc, output_dir)
    save_probability_plot(test["y"], test["prob"], output_dir)


def save_selected_measurements(selected_idx, selected_labels, output_dir):
    path = Path(output_dir) / "selected_measurements.txt"
    with path.open("w") as f:
        for idx, label in zip(selected_idx, selected_labels):
            f.write(f"{idx:4d}  {label}\n")


def save_checkpoint(
    model,
    mean,
    std,
    selected_idx,
    selected_labels,
    K,
    args,
    output_dir,
):
    checkpoint = {
        "state_dict": model.state_dict(),
        "K": K,
        "mean": torch.tensor(mean, dtype=torch.float32),
        "std": torch.tensor(std, dtype=torch.float32),
        "pauli_labels": PAULI_LABELS,
        "selected_indices": selected_idx.tolist(),
        "selected_labels": selected_labels,
        "N": N,
        "N_exc_initial": N_EXC_INITIAL,
        "eps_C": EPS_C,
        "eps_R2": EPS_R2,
        "conditioning": "C_lin < EPS_C",
        "label_after_conditioning": "R2 > EPS_R2",
        "match_eps_pout": args.match_eps_pout,
        "match_eps_varq": args.match_eps_varq,
        "match_eps_qmean": args.match_eps_qmean,
        "targeted_sampling": True,
        "regime_a_fraction": args.regime_a_fraction,
        "late_time_fraction": args.late_time_fraction,
        "late_time_start": args.late_time_start,
    }
    torch.save(checkpoint, Path(output_dir) / "model_best.pt")


def save_matched_dataset(data, train_idx, val_idx, test_idx, output_dir):
    payload = {
        key: torch.tensor(value) if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    payload |= {
        "pauli_labels": PAULI_LABELS,
        "train_idx": torch.tensor(train_idx),
        "val_idx": torch.tensor(val_idx),
        "test_idx": torch.tensor(test_idx),
    }
    torch.save(payload, Path(output_dir) / "dataset_matched.pt")


def charge_baselines(data, test_idx):
    y = data["y"][test_idx].astype(int)
    q_auc = orientation_free_auc(y, data["q_mean"][test_idx])
    var_auc = orientation_free_auc(y, data["var_q"][test_idx])
    pout_auc = orientation_free_auc(y, data["p_out"][test_idx])
    return q_auc, var_auc, pout_auc


def write_summary(
    args,
    raw,
    matched,
    match_info,
    best_val_auc,
    test,
    charge_aucs,
    selected_labels,
    output_dir,
):
    q_auc, var_auc, pout_auc = charge_aucs
    with (Path(output_dir) / "summary.txt").open("w") as f:
        f.write("N=6 continuously charge-matched SW-like learning\n\n")
        f.write(f"N = {N}\n")
        f.write(f"initial N_exc = {N_EXC_INITIAL}\n")
        f.write(f"candidate observables = {len(PAULI_LABELS)} (1-/2-body Pauli)\n")
        f.write(f"K = {args.K}\n")
        f.write(f"raw trajectories = {args.n_trajectories}\n")
        f.write(f"times/trajectory = {args.n_times}\n")
        f.write(f"raw states = {len(raw['y'])}\n")
        f.write(f"raw R2-positive fraction = {raw['y'].mean():.8f}\n\n")

        f.write("Targeted sampling\n")
        f.write(f"  regime A fraction = {args.regime_a_fraction}\n")
        f.write("  regime A: gamma_gain in [0, 0.025], gamma_loss in [0.15, 0.35]\n")
        f.write("  regime B: gamma_gain in [0.025, 0.18], gamma_loss in [0.02, 0.25]\n")
        f.write(f"  late-time fraction = {args.late_time_fraction}\n")
        f.write(f"  late-time start = {args.late_time_start}\n\n")

        f.write("Conditioning\n")
        f.write(f"  keep C_lin < {EPS_C}\n")
        f.write(f"  conditioned states = {match_info['n_conditioned']}\n")
        f.write(
            f"  conditioned class0 (R2 <= {EPS_R2}) = "
            f"{match_info['conditioned_class0']}\n"
        )
        f.write(
            f"  conditioned class1 (R2 > {EPS_R2}) = "
            f"{match_info['conditioned_class1']}\n"
        )
        f.write(
            f"  conditioned positive fraction = "
            f"{match_info['conditioned_positive_fraction']:.8f}\n\n"
        )

        f.write("Continuous charge matching after conditioning\n")
        f.write(f"  max |Delta p_out| = {args.match_eps_pout}\n")
        f.write(f"  max |Delta Var(Q)| = {args.match_eps_varq}\n")
        f.write(f"  max |Delta <Q>| = {args.match_eps_qmean}\n")
        f.write(f"  train pairs = {match_info['train']['pairs']}\n")
        f.write(f"  val pairs = {match_info['val']['pairs']}\n")
        f.write(f"  test pairs = {match_info['test']['pairs']}\n")
        f.write(f"  matched states = {len(matched['y'])}\n")
        f.write(f"  positive fraction = {matched['y'].mean():.8f}\n\n")

        f.write("Frozen-test performance\n")
        f.write(f"  best val AUC = {best_val_auc:.8f}\n")
        f.write(f"  test AUC = {test['auc']:.8f}\n")
        f.write(f"  test ACC = {test['acc']:.8f}\n\n")

        f.write("Charge-only orientation-free AUC on SAME test set\n")
        f.write(f"  <Q> = {q_auc:.8f}\n")
        f.write(f"  Var(Q) = {var_auc:.8f}\n")
        f.write(f"  p_out = {pout_auc:.8f}\n\n")

        f.write("Selected observables\n")
        for label in selected_labels:
            f.write(f"  {label}\n")


# ---------------------------------------------------------------------------
# Final selected-initial-state demonstration
# ---------------------------------------------------------------------------

def ideal_demo_parameters():
    """
    Clean, homogeneous member of the training family.
    It includes gain/loss, because sector mixing is required for the SW-like
    strong-to-weak crossover.
    """
    return dict(
        Jxy=np.ones(N - 1),
        Jz=0.8 * np.ones(N - 1),
        h=np.zeros(N),
        gamma_phi=0.08,
        gamma_hop=0.08,
        gamma_loss=0.24,
        gamma_gain=0.012,
        regime="idealized",
    )


def realistic_demo_parameters(seed):
    """
    One fixed disordered member of the same family used to generate training
    trajectories. It is deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    pars = sample_targeted_parameters(rng, regime_a_fraction=0.5)
    pars["regime"] = "realistic"
    return pars


@torch.no_grad()
def witness_probability(model, rho, mean, std):
    features = local_pauli_features(rho)[None, :]
    x = (features - np.asarray(mean).reshape(1, -1)) / np.asarray(std).reshape(1, -1)
    xt = torch.tensor(x, dtype=torch.float32)
    logits, _ = model(xt)
    return float(torch.sigmoid(logits)[0].cpu())


def evaluate_demo_trajectory(model, mean, std, rho0, pars, times):
    H = build_hamiltonian(pars["Jxy"], pars["Jz"], pars["h"])
    L = liouvillian(
        H,
        pars["gamma_phi"],
        pars["gamma_hop"],
        pars["gamma_loss"],
        pars["gamma_gain"],
    )
    states = evolve_at_times(L, rho0, times)

    linear = np.array([linear_correlator(rho) for rho in states])
    r2 = np.array([renyi2_correlator(rho) for rho in states])
    witness = np.array([witness_probability(model, rho, mean, std) for rho in states])
    q = np.array([charge_diagnostics(rho)[0] for rho in states])
    varq = np.array([charge_diagnostics(rho)[1] for rho in states])
    pout = np.array([charge_diagnostics(rho)[2] for rho in states])

    return {
        "time": np.asarray(times),
        "linear": linear,
        "r2": r2,
        "witness": witness,
        "q_mean": q,
        "var_q": varq,
        "p_out": pout,
    }


def save_demo_plot(result, name, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)

    axes[0].plot(result["time"], result["linear"], label=r"$C_{\rm lin}$")
    axes[0].axhline(EPS_C, linestyle="--", label=r"$\epsilon_C$")
    axes[0].set_ylabel(r"$C_{\rm lin}$")
    axes[0].legend()

    axes[1].plot(result["time"], result["r2"], label=r"$R_2$")
    axes[1].axhline(EPS_R2, linestyle="--", label=r"$\epsilon_{R_2}$")
    axes[1].set_ylabel(r"$R_2$")
    axes[1].legend()

    axes[2].plot(result["time"], result["witness"], label=r"$P_{\rm SW}$")
    axes[2].axhline(0.5, linestyle="--", label="decision threshold")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel(r"$P_{\rm SW}$")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend()

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"demo_{name}.png", dpi=180)
    plt.close(fig)


def save_demo_csv(result, name, output_dir):
    arr = np.column_stack(
        [
            result["time"],
            result["linear"],
            result["r2"],
            result["witness"],
            result["q_mean"],
            result["var_q"],
            result["p_out"],
        ]
    )
    header = "time,C_lin,R2,P_SW,Q_mean,VarQ,p_out"
    np.savetxt(
        Path(output_dir) / f"demo_{name}.csv",
        arr,
        delimiter=",",
        header=header,
        comments="",
    )


def run_final_demo(
    model,
    mean,
    std,
    output_dir,
    t_max=10.0,
    n_times=101,
    bitstring="101010",
    seed=DEFAULT_SEED + 100,
):
    """
    Final sanity check on one chosen Q=3 initial basis state.

    idealized:
        homogeneous Hamiltonian + fixed clean Lindblad rates.

    realistic:
        one disordered trajectory drawn from the training parameter family.

    Saves linear correlator, R2 and learned witness versus time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rho0 = basis_state_rho(bitstring)
    times = np.linspace(0.0, t_max, n_times)

    cases = {
        "idealized": ideal_demo_parameters(),
        "realistic": realistic_demo_parameters(seed),
    }

    results = {}
    for name, pars in cases.items():
        result = evaluate_demo_trajectory(model, mean, std, rho0, pars, times)
        results[name] = result
        save_demo_plot(result, name, output_dir)
        save_demo_csv(result, name, output_dir)

    with (Path(output_dir) / "demo_parameters.txt").open("w") as f:
        f.write(f"initial basis state = |{bitstring}>\n\n")
        for name, pars in cases.items():
            f.write(f"[{name}]\n")
            f.write(f"Jxy = {np.asarray(pars['Jxy']).tolist()}\n")
            f.write(f"Jz = {np.asarray(pars['Jz']).tolist()}\n")
            f.write(f"h = {np.asarray(pars['h']).tolist()}\n")
            for key in ("gamma_phi", "gamma_hop", "gamma_loss", "gamma_gain"):
                f.write(f"{key} = {pars[key]}\n")
            f.write("\n")

    return results
