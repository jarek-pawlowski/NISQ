import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# Fixed observable-selection benchmark for physics-informed QRC
#
# We compare fixed oracle-best subsets against a learned v5-style selector.
# We use all 10 measurement times: t = 0,5,...,45.
# Each policy selects exactly 4 of the 15 candidate observables.
#
# For each hardware/noise condition:
#   - generate one common reservoir dataset with ALL 15 observables,
#   - slice the same data with several fixed 4-observable policies,
#   - train a separate PhysicsReadout for every policy,
#   - compare test MSE.
#
# If the best fixed observable subset depends on the hardware condition,
# adaptive observable selection has a concrete task-level motivation.
# ============================================================

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# 1. 4-spin operators
# ------------------------------------------------------------

N = 4
DIM = 2**N

I2 = torch.eye(2, dtype=torch.complex128, device=device)
X = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128, device=device)
Y = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex128, device=device)
Z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128, device=device)
I16 = torch.eye(DIM, dtype=torch.complex128, device=device)


def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = torch.kron(out, op)
    return out


def op1(A, i):
    ops = [I2 for _ in range(N)]
    ops[i] = A
    return kron_all(ops)


def op2(A, i, B, j):
    ops = [I2 for _ in range(N)]
    ops[i] = A
    ops[j] = B
    return kron_all(ops)


Xq = [op1(X, i) for i in range(N)]
Yq = [op1(Y, i) for i in range(N)]
Zq = [op1(Z, i) for i in range(N)]
XX = [op2(X, i, X, i + 1) for i in range(N - 1)]
YY = [op2(Y, i, Y, i + 1) for i in range(N - 1)]
ZZ = [op2(Z, i, Z, i + 1) for i in range(N - 1)]

Y0 = Yq[0]

observables = []
observable_names = []
observable_qubit = []

for i in range(N):
    observables += [Xq[i], Yq[i], Zq[i]]
    observable_names += [f"X{i}", f"Y{i}", f"Z{i}"]
    observable_qubit += [i, i, i]

for i in range(N - 1):
    observables.append(ZZ[i])
    observable_names.append(f"Z{i}Z{i+1}")
    # Keep ZZ correlations untouched by the extra local readout-noise
    # stress test, matching the oracle scan used to define the fixed sets.
    observable_qubit.append(None)

NOBS = len(observables)
name_to_idx = {name: i for i, name in enumerate(observable_names)}

print("candidate observables:", observable_names)

# ------------------------------------------------------------
# 2. Same signal data for ALL tests
# ------------------------------------------------------------

T = 50
times = torch.linspace(0.0, 2.0 * math.pi, T, device=device)
NFREQ = T // 2 + 1


def generate_signals(n):
    A1 = 0.4 + 0.6 * torch.rand(n, device=device)
    A2 = 0.2 + 0.5 * torch.rand(n, device=device)
    w1 = 0.7 + 1.3 * torch.rand(n, device=device)
    w2 = 2.2 + 1.5 * torch.rand(n, device=device)
    p1 = 2.0 * math.pi * torch.rand(n, device=device)
    p2 = 2.0 * math.pi * torch.rand(n, device=device)

    t = times[None, :]
    return (
        A1[:, None] * torch.sin(w1[:, None] * t + p1[:, None])
        + A2[:, None] * torch.sin(w2[:, None] * t + p2[:, None])
    )


NTRAIN_TOTAL = 1000
NVAL = 200
NTEST = 250

all_train_u = generate_signals(NTRAIN_TOTAL)
train_u = all_train_u[:-NVAL]
val_u = all_train_u[-NVAL:]
test_u = generate_signals(NTEST)

print("train:", train_u.shape, "val:", val_u.shape, "test:", test_u.shape)

# ------------------------------------------------------------
# 3. ALL ten measurement times
# ------------------------------------------------------------

SAMPLE_EVERY = 10  # 5
candidate_times = list(range(0, T, SAMPLE_EVERY))
NTIME = len(candidate_times)

assert NTIME == 5  # 10
print("measurement times:", candidate_times)

# ------------------------------------------------------------
# 4. Fixed reservoir -- same as adaptive_pi_qrc_v5
# ------------------------------------------------------------

J_FIXED = torch.tensor([1.00, 0.83, 1.17], device=device)
H_FIXED = torch.tensor([0.21, -0.13, 0.08, 0.17], device=device)
DT_FIXED = torch.tensor(0.35, device=device)
ALPHA_FIXED = torch.tensor(0.80, device=device)


def build_H():
    H = torch.zeros((DIM, DIM), dtype=torch.complex128, device=device)

    for i in range(N - 1):
        H = H + J_FIXED[i] * (XX[i] + YY[i] + ZZ[i])

    for i in range(N):
        H = H + H_FIXED[i] * Zq[i]

    return H


H_RES = build_H()
URES = torch.matrix_exp(-1j * H_RES * DT_FIXED)

# ------------------------------------------------------------
# 5. Same heterogeneous noise model as before
# ------------------------------------------------------------

def embed_single_qubit_matrix(A, site):
    ops = [I2 for _ in range(N)]
    ops[site] = A
    return kron_all(ops)


def make_noise_operators(gamma_phi, gamma_1):
    gamma_phi = torch.as_tensor(
        gamma_phi, dtype=torch.get_default_dtype(), device=device
    )
    gamma_1 = torch.as_tensor(
        gamma_1, dtype=torch.get_default_dtype(), device=device
    )

    p_phi = 0.5 * (1.0 - torch.exp(-2.0 * gamma_phi * DT_FIXED))
    p_1 = 1.0 - torch.exp(-gamma_1 * DT_FIXED)

    p_phi = torch.clamp(p_phi, 0.0, 0.499999)
    p_1 = torch.clamp(p_1, 0.0, 0.999999)

    K0, K1 = [], []

    for i in range(N):
        pi = p_1[i]

        K0_1q = torch.stack([
            torch.stack([
                torch.ones((), dtype=torch.complex128, device=device),
                torch.zeros((), dtype=torch.complex128, device=device),
            ]),
            torch.stack([
                torch.zeros((), dtype=torch.complex128, device=device),
                torch.sqrt(1.0 - pi).to(torch.complex128),
            ])
        ])

        K1_1q = torch.stack([
            torch.stack([
                torch.zeros((), dtype=torch.complex128, device=device),
                torch.sqrt(pi).to(torch.complex128),
            ]),
            torch.stack([
                torch.zeros((), dtype=torch.complex128, device=device),
                torch.zeros((), dtype=torch.complex128, device=device),
            ])
        ])

        K0.append(embed_single_qubit_matrix(K0_1q, i))
        K1.append(embed_single_qubit_matrix(K1_1q, i))

    return p_phi, K0, K1


def apply_noise(rho, p_phi, K0, K1):
    for i in range(N):
        Zi = Zq[i]

        rho = (
            (1.0 - p_phi[i]) * rho
            + p_phi[i] * (Zi @ rho @ Zi)
        )

        k0 = K0[i]
        k1 = K1[i]

        rho = (
            k0 @ rho @ k0.conj().T
            + k1 @ rho @ k1.conj().T
        )

    return rho

# ------------------------------------------------------------
# 6. Compute ALL 15 observables, at ALL 10 times, once
# ------------------------------------------------------------

def initial_density_matrix(batch_size):
    psi0 = torch.zeros(DIM, dtype=torch.complex128, device=device)
    psi0[0] = 1.0
    rho0 = torch.outer(psi0, psi0.conj())
    return rho0[None, :, :].repeat(batch_size, 1, 1)


def expectation_batch(rho, O):
    return torch.real(torch.einsum("bij,ji->b", rho, O))


@torch.no_grad()
def reservoir_all_features(
    signals,
    gamma_phi,
    gamma_1,
    nshots,
    add_shot_noise=True,
    measurement_noise_qubit=None,
    measurement_noise_std=0.0,
):
    """
    Returns tensor [B, 10, 15].

    Every fixed policy below is only a slice of this same tensor.
    This makes the comparison fair: policies see exactly the same
    signal samples and the same shot-noise realization.
    """
    B = signals.shape[0]

    p_phi, K0, K1 = make_noise_operators(gamma_phi, gamma_1)
    rho = initial_density_matrix(B)

    features = []

    for n in range(T):
        angle = ALPHA_FIXED * signals[:, n]

        c = torch.cos(0.5 * angle)[:, None, None].to(torch.complex128)
        s = torch.sin(0.5 * angle)[:, None, None].to(torch.complex128)

        Uin = c * I16[None, :, :] - 1j * s * Y0[None, :, :]

        rho = Uin @ rho @ Uin.conj().transpose(-1, -2)
        rho = URES @ rho @ URES.conj().T
        rho = apply_noise(rho, p_phi, K0, K1)

        if n % SAMPLE_EVERY == 0:
            vals = torch.stack(
                [expectation_batch(rho, O) for O in observables],
                dim=1
            )

            if add_shot_noise:
                var = torch.clamp(1.0 - vals**2, min=1e-10)
                std = torch.sqrt(var / float(nshots))
                vals = vals + torch.randn_like(vals) * std

            # Additional qubit-local measurement/readout noise.
            # This does NOT modify the reservoir dynamics; it only corrupts
            # local X/Y/Z measurements belonging to the selected qubit.
            if measurement_noise_qubit is not None and measurement_noise_std > 0.0:
                for k, q in enumerate(observable_qubit):
                    if q == measurement_noise_qubit:
                        vals[:, k] = (
                            vals[:, k]
                            + measurement_noise_std * torch.randn_like(vals[:, k])
                        )

            features.append(vals)

    return torch.stack(features, dim=1)

# ------------------------------------------------------------
# 7. Fixed oracle-best 4-observable policies
# ------------------------------------------------------------

# These sets are copied from the previous exhaustive oracle scan.
# This script intentionally DOES NOT recompute the oracle.
#
# All policies have exactly the same resource cost:
#     4 observables x all 10 measurement times = 40 values.
#
# The same fixed oracle sets are cross-tested on every hardware condition.
# We later add one extra, condition-dependent policy produced by the
# learned v5-style selector.

MEASUREMENT_POLICIES = {
    "XY_near_q0":  ["X0", "Y0", "X1", "Y1"],
    "XY_middle":   ["X1", "Y1", "X2", "Y2"],
    "XY_far":      ["X2", "Y2", "X3", "Y3"],
    "X_chain":     ["X0", "X1", "X2", "X3"],
    "Y_chain":     ["Y0", "Y1", "Y2", "Y3"],
    "Z_chain":     ["Z0", "Z1", "Z2", "Z3"],
    "Z_corr":      ["Z0", "Z3", "Z0Z1", "Z2Z3"]
}

def select_policy_features(R_all, obs_names):
    """
    R_all: [B, 10, 15]
    output: [B, 40]
    """
    idx = [name_to_idx[name] for name in obs_names]
    return R_all[:, :, idx].flatten(start_dim=1)

# ------------------------------------------------------------
# 8. Four hardware cases
# ------------------------------------------------------------

CONDITIONS = {
    "clean": {
        "gamma_phi": [0.0, 0.0, 0.0, 0.0],
        "gamma_1":   [0.0, 0.0, 0.0, 0.0],
        "nshots": 10000,
        "measurement_noise_qubit": None,
        "measurement_noise_std": 0.0,
    },

    "bad_q0_z": {
        "gamma_phi": [0.30, 0.01, 0.01, 0.01],
        "gamma_1":   [0.12, 0.005, 0.005, 0.005],
        "nshots": 1000,
        "measurement_noise_qubit": None,
        "measurement_noise_std": 0.0,
    },

    "bad_q2_z": {
        "gamma_phi": [0.01, 0.01, 0.30, 0.01],
        "gamma_1":   [0.005, 0.005, 0.12, 0.005],
        "nshots": 1000,
        "measurement_noise_qubit": None,
        "measurement_noise_std": 0.0,
    },

    # Readout-noise stress tests: reservoir dynamics are only mildly noisy,
    # but local X/Y/Z measurements on one qubit are strongly corrupted.
    "bad_q0_meas": {
        "gamma_phi": [0.02, 0.02, 0.02, 0.02],
        "gamma_1":   [0.005, 0.005, 0.005, 0.005],
        "nshots": 1000,
        "measurement_noise_qubit": 0,
        "measurement_noise_std": 0.35,
    },

    "bad_q2_meas": {
        "gamma_phi": [0.02, 0.02, 0.02, 0.02],
        "gamma_1":   [0.005, 0.005, 0.005, 0.005],
        "nshots": 1000,
        "measurement_noise_qubit": 2,
        "measurement_noise_std": 0.35,
    },

    # Keep the original strongly heterogeneous case as an extrapolation test.
    "noisy": {
        "gamma_phi": [0.25, 0.22, 0.30, 0.20],
        "gamma_1":   [0.10, 0.08, 0.12, 0.07],
        "nshots": 300,
        "measurement_noise_qubit": None,
        "measurement_noise_std": 0.0,
    },
}

# ------------------------------------------------------------
# 9. Physics-informed readout
# ------------------------------------------------------------

class PhysicsReadout(nn.Module):
    def __init__(self, nin):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(nin, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2 * NFREQ)
        )

    def forward(self, R):
        x = self.net(R)

        F = torch.complex(
            x[:, :NFREQ],
            x[:, NFREQ:]
        )

        u_hat = torch.fft.irfft(F, n=T, dim=1)
        return F, u_hat


mse = nn.MSELoss()

# ------------------------------------------------------------
# 10. Reader training
# ------------------------------------------------------------

MAX_EPOCHS = 2500
PATIENCE = 300
LR = 2e-3
WEIGHT_DECAY = 1e-4

# Set to 1 for a fast first run; 3 is better for a stable comparison.
N_READOUT_SEEDS = 3


def train_readout(R_train, u_train, R_val, u_val, seed=0):
    torch.manual_seed(seed)

    model = PhysicsReadout(R_train.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1000, 1800],
        gamma=0.3
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(MAX_EPOCHS):
        model.train()

        optimizer.zero_grad()

        _, pred = model(R_train)
        loss = mse(pred, u_train)

        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()

        with torch.no_grad():
            _, pred_val = model(R_val)
            val_loss = mse(pred_val, u_val).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            stale += 1

        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch

# ------------------------------------------------------------
# 11. Cache the reservoir data
# ------------------------------------------------------------

print("\nGenerating common reservoir datasets...")

CACHE = {}

for ic, (condition_name, cfg) in enumerate(CONDITIONS.items()):
    print(" ", condition_name)

    # Fixed seeds make within-condition policy comparisons share
    # exactly the same shot-noise realization.
    torch.manual_seed(SEED + 1000 * ic + 1)
    R_train_all = reservoir_all_features(
        train_u,
        cfg["gamma_phi"],
        cfg["gamma_1"],
        cfg["nshots"],
        add_shot_noise=True,
        measurement_noise_qubit=cfg["measurement_noise_qubit"],
        measurement_noise_std=cfg["measurement_noise_std"],
    )

    torch.manual_seed(SEED + 1000 * ic + 2)
    R_val_all = reservoir_all_features(
        val_u,
        cfg["gamma_phi"],
        cfg["gamma_1"],
        cfg["nshots"],
        add_shot_noise=True,
        measurement_noise_qubit=cfg["measurement_noise_qubit"],
        measurement_noise_std=cfg["measurement_noise_std"],
    )

    torch.manual_seed(SEED + 1000 * ic + 3)
    R_test_all = reservoir_all_features(
        test_u,
        cfg["gamma_phi"],
        cfg["gamma_1"],
        cfg["nshots"],
        add_shot_noise=True,
        measurement_noise_qubit=cfg["measurement_noise_qubit"],
        measurement_noise_std=cfg["measurement_noise_std"],
    )

    CACHE[condition_name] = {
        "train": R_train_all,
        "val": R_val_all,
        "test": R_test_all,
    }



# ------------------------------------------------------------
# 11b. Adaptive selector trained exactly in the adaptive_pi_qrc style
# ------------------------------------------------------------
#
# Difference from the old benchmark selector:
#   - one AuxiliaryController conditions on the full hardware/noise context,
#   - it jointly selects observables AND times,
#   - selector + PhysicsReadout are trained end-to-end,
#   - soft-budget warm-up -> Gumbel straight-through Top-K,
#   - evaluation uses the SAME jointly trained PhysicsReadout.
#
# The fixed benchmark policies below are still trained independently, exactly
# as before.  Thus this script compares:
#
#   fixed policy + freshly trained readout
#             versus
#   adaptive controller + its jointly trained readout.
#
# NOTE ON RESOURCE COST:
#   fixed policies use K_OBS=4 observables at ALL NTIME benchmark times;
#   adaptive policy uses K_OBS=4 observables at K_TIME=2 selected times.
# ------------------------------------------------------------

K_OBS = 4
K_TIME = 5

GAMMA_PHI_MAX = 0.30
GAMMA_1_MAX = 0.12
MEAS_NOISE_MAX = 0.35

# task + gamma_phi[4] + gamma_1[4] + measurement_noise[4] + nshots
CONTEXT_DIM = 1 + N + N + N + 1
NR_ADAPTIVE = NTIME * NOBS
READOUT_DIM_ADAPTIVE = NR_ADAPTIVE + NOBS + NTIME


def measurement_noise_vector(cfg):
    v = torch.zeros(N, device=device)
    q = cfg["measurement_noise_qubit"]

    if q is not None:
        v[int(q)] = float(cfg["measurement_noise_std"])

    return v


def controller_context(cfg, task_id=1.0):
    gp = torch.tensor(
        cfg["gamma_phi"],
        dtype=torch.get_default_dtype(),
        device=device
    ) / GAMMA_PHI_MAX

    g1 = torch.tensor(
        cfg["gamma_1"],
        dtype=torch.get_default_dtype(),
        device=device
    ) / GAMMA_1_MAX

    gm = measurement_noise_vector(cfg) / MEAS_NOISE_MAX

    nshots = torch.tensor(
        float(cfg["nshots"]),
        dtype=torch.get_default_dtype(),
        device=device
    )

    ns = (torch.log10(nshots) - 2.0) / 2.0

    task = torch.tensor(
        [task_id],
        dtype=torch.get_default_dtype(),
        device=device
    )

    return torch.cat([
        task,
        gp,
        g1,
        gm,
        ns.reshape(1),
    ])[None, :]


class AuxiliaryController(nn.Module):
    """
    Same policy parameterization as adaptive_pi_qrc.py:

        logits = learned global baseline
                 + policy_beta * hardware-dependent correction.
    """

    def __init__(self, hidden=64, policy_beta=4.0):
        super().__init__()

        self.policy_beta = policy_beta

        self.backbone = nn.Sequential(
            nn.Linear(CONTEXT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Kept for architectural compatibility with adaptive_pi_qrc.py.
        # Reservoir controls remain fixed in this benchmark.
        self.head_physics = nn.Linear(hidden, 3 + 4 + 1 + 1)

        self.base_obs = nn.Parameter(torch.zeros(NOBS))
        self.base_time = nn.Parameter(torch.zeros(NTIME))

        self.obs_correction = nn.Linear(hidden, NOBS)
        self.time_correction = nn.Linear(hidden, NTIME)

        nn.init.normal_(self.obs_correction.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.obs_correction.bias)

        nn.init.normal_(self.time_correction.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.time_correction.bias)

    def forward(self, context):
        z = self.backbone(context)

        delta_obs = self.obs_correction(z).squeeze(0)
        delta_time = self.time_correction(z).squeeze(0)

        logits_obs = self.base_obs + self.policy_beta * delta_obs
        logits_time = self.base_time + self.policy_beta * delta_time

        return {
            "logits_obs": logits_obs,
            "logits_time": logits_time,
            "delta_obs": delta_obs,
            "delta_time": delta_time,
        }


def allocation_scores(logits, tau=1.0):
    return torch.softmax(logits / tau, dim=-1)


def soft_budget_gate(logits, k, tau=1.0):
    soft = k * torch.softmax(logits / tau, dim=-1)
    return soft


def straight_through_topk(logits, k, tau=1.0, training=True):
    if training:
        eps = 1e-10
        u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        gumbel = -torch.log(-torch.log(u))
        scores = logits + gumbel
    else:
        scores = logits

    soft = k * torch.softmax(scores / tau, dim=-1)

    idx = torch.topk(
        scores,
        k=k,
        dim=-1
    ).indices

    hard = torch.zeros_like(logits)
    hard[idx] = 1.0

    if training:
        gate = hard.detach() - soft.detach() + soft
    else:
        gate = hard

    return gate, hard


class AdaptivePhysicsReadout(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(READOUT_DIM_ADAPTIVE, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2 * NFREQ),
        )

    def forward(self, x):
        y = self.net(x)

        F = torch.complex(
            y[:, :NFREQ],
            y[:, NFREQ:]
        )

        u_hat = torch.fft.irfft(
            F,
            n=T,
            dim=1
        )

        return F, u_hat


def build_adaptive_readout_input(
    R_all,
    gate_obs,
    gate_time
):
    """
    R_all: [B, NTIME, NOBS]

    Same information flow as adaptive_pi_qrc.py:
      - apply observable mask,
      - apply time mask,
      - flatten all candidate slots,
      - append explicit observable/time metadata.
    """
    R = (
        R_all
        * gate_obs[None, None, :]
        * gate_time[None, :, None]
    )

    R = R.flatten(start_dim=1)

    B = R.shape[0]

    obs_meta = gate_obs[None, :].expand(B, -1)
    time_meta = gate_time[None, :].expand(B, -1)

    return torch.cat(
        [R, obs_meta, time_meta],
        dim=1
    )


def fresh_reservoir_features(
    signals,
    cfg
):
    """
    Generate a fresh noisy reservoir realization for the current minibatch.

    This deliberately does NOT reuse CACHE during adaptive training, so shot
    noise/readout noise are resampled from epoch to epoch, matching the
    adaptive_pi_qrc training logic more closely.
    """
    return reservoir_all_features(
        signals,
        cfg["gamma_phi"],
        cfg["gamma_1"],
        cfg["nshots"],
        add_shot_noise=True,
        measurement_noise_qubit=cfg["measurement_noise_qubit"],
        measurement_noise_std=cfg["measurement_noise_std"],
    )


controller = AuxiliaryController(
    hidden=64,
    policy_beta=4.0
).to(device)

adaptive_readout = AdaptivePhysicsReadout().to(device)


# Same training structure/hyperparameters as adaptive_pi_qrc.py.
ADAPTIVE_EPOCHS = 4000
ADAPTIVE_SIGNALS_PER_CONDITION = 8

ADAPTIVE_SOFT_WARMUP_EPOCHS = 600
ADAPTIVE_ENTROPY_WARMUP_EPOCHS = 400

ADAPTIVE_TAU_START = 2.0
ADAPTIVE_TAU_END = 0.75
ADAPTIVE_TAU_ANNEAL_EPOCHS = 2000

ADAPTIVE_LAMBDA_ENTROPY = 2e-3

adaptive_optimizer = torch.optim.AdamW(
    list(controller.parameters())
    + list(adaptive_readout.parameters()),
    lr=2e-3,
    weight_decay=1e-4,
)

adaptive_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    adaptive_optimizer,
    milestones=[1800, 3000],
    gamma=0.3,
)

adaptive_history_task = []
adaptive_history_total = []
adaptive_history_delta = []

condition_items = list(CONDITIONS.items())

print("\nTraining adaptive selector + readout end-to-end...")

for epoch in range(ADAPTIVE_EPOCHS):
    controller.train()
    adaptive_readout.train()

    tau_frac = min(
        epoch / max(ADAPTIVE_TAU_ANNEAL_EPOCHS - 1, 1),
        1.0
    )

    tau = (
        ADAPTIVE_TAU_START
        * (ADAPTIVE_TAU_END / ADAPTIVE_TAU_START) ** tau_frac
    )

    soft_phase = (
        epoch < ADAPTIVE_SOFT_WARMUP_EPOCHS
    )

    entropy_phase = (
        epoch < ADAPTIVE_ENTROPY_WARMUP_EPOCHS
    )

    total_task = 0.0
    total_entropy = 0.0
    total_delta = 0.0

    # Exactly as in adaptive_pi_qrc.py:
    # each optimizer step sees all explicit hardware conditions.
    for condition_name, cfg in condition_items:
        theta = controller(
            controller_context(cfg)
        )

        if soft_phase:
            gate_obs = soft_budget_gate(
                theta["logits_obs"],
                K_OBS,
                tau=tau
            )

            gate_time = soft_budget_gate(
                theta["logits_time"],
                K_TIME,
                tau=tau
            )

        else:
            gate_obs, _ = straight_through_topk(
                theta["logits_obs"],
                K_OBS,
                tau=tau,
                training=True
            )

            gate_time, _ = straight_through_topk(
                theta["logits_time"],
                K_TIME,
                tau=tau,
                training=True
            )

        ids = torch.randint(
            0,
            train_u.shape[0],
            (ADAPTIVE_SIGNALS_PER_CONDITION,),
            device=device
        )

        u = train_u[ids]

        R_all = fresh_reservoir_features(
            u,
            cfg
        )

        x = build_adaptive_readout_input(
            R_all,
            gate_obs,
            gate_time
        )

        _, u_hat = adaptive_readout(x)

        task_loss = mse(
            u_hat,
            u
        )

        q_obs = allocation_scores(
            theta["logits_obs"],
            tau=1.0
        )

        q_time = allocation_scores(
            theta["logits_time"],
            tau=1.0
        )

        entropy = (
            -(q_obs * torch.log(q_obs + 1e-12)).sum()
            -(q_time * torch.log(q_time + 1e-12)).sum()
        )

        entropy_loss = -entropy

        delta_norm = (
            theta["delta_obs"].pow(2).mean()
            + theta["delta_time"].pow(2).mean()
        )

        total_task = total_task + task_loss
        total_entropy = total_entropy + entropy_loss
        total_delta = total_delta + delta_norm

    nc = len(condition_items)

    total_task = total_task / nc
    total_entropy = total_entropy / nc
    total_delta = total_delta / nc

    loss = total_task

    if entropy_phase:
        loss = (
            loss
            + ADAPTIVE_LAMBDA_ENTROPY
            * total_entropy
        )

    adaptive_optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(controller.parameters())
        + list(adaptive_readout.parameters()),
        max_norm=5.0
    )

    adaptive_optimizer.step()
    adaptive_scheduler.step()

    adaptive_history_task.append(
        total_task.item()
    )

    adaptive_history_total.append(
        loss.item()
    )

    adaptive_history_delta.append(
        total_delta.item()
    )

    if epoch % 50 == 0:
        phase = (
            "soft"
            if soft_phase
            else "gumbel-topk"
        )

        print(
            f"{epoch:4d} | "
            f"phase={phase:11s} | "
            f"task={total_task.item():.5e} | "
            f"delta={total_delta.item():.3e} | "
            f"tau={tau:.3f} | "
            f"lr={adaptive_optimizer.param_groups[0]['lr']:.2e}"
        )


@torch.no_grad()
def evaluate_adaptive_condition(
    cfg,
    signals
):
    """
    Deterministic hard Top-K evaluation using the SAME jointly trained
    adaptive readout, exactly as in adaptive_pi_qrc.py.
    """
    controller.eval()
    adaptive_readout.eval()

    theta = controller(
        controller_context(cfg)
    )

    _, hard_obs = straight_through_topk(
        theta["logits_obs"],
        K_OBS,
        tau=ADAPTIVE_TAU_END,
        training=False
    )

    _, hard_time = straight_through_topk(
        theta["logits_time"],
        K_TIME,
        tau=ADAPTIVE_TAU_END,
        training=False
    )

    R_all = fresh_reservoir_features(
        signals,
        cfg
    )

    x = build_adaptive_readout_input(
        R_all,
        hard_obs,
        hard_time
    )

    _, u_hat = adaptive_readout(x)

    err = mse(
        u_hat,
        signals
    ).item()

    selected_obs = [
        observable_names[i]
        for i in range(NOBS)
        if hard_obs[i].item() > 0.5
    ]

    selected_times = [
        candidate_times[i]
        for i in range(NTIME)
        if hard_time[i].item() > 0.5
    ]

    return {
        "mse": err,
        "u_hat": u_hat.detach().cpu(),
        "selected_obs": selected_obs,
        "selected_times": selected_times,
        "hard_obs": hard_obs.detach().cpu(),
        "hard_time": hard_time.detach().cpu(),
        "prob_obs": allocation_scores(
            theta["logits_obs"]
        ).detach().cpu(),
        "prob_time": allocation_scores(
            theta["logits_time"]
        ).detach().cpu(),
    }


# Evaluate adaptive policy several times to estimate variation caused by
# shot/readout noise. The neural weights are NOT retrained between repeats.
ADAPTIVE_TEST_REPS = 3
ADAPTIVE_RESULTS = {}

print("\nAdaptive selector test:")

for ic, (condition_name, cfg) in enumerate(condition_items):
    errs = []
    first = None

    for r in range(ADAPTIVE_TEST_REPS):
        torch.manual_seed(
            SEED + 50000 + 1000 * ic + r
        )

        out = evaluate_adaptive_condition(
            cfg,
            test_u
        )

        errs.append(
            out["mse"]
        )

        if first is None:
            first = out

    ADAPTIVE_RESULTS[condition_name] = {
        **first,
        "mean_mse": float(np.mean(errs)),
        "std_mse": (
            float(np.std(errs, ddof=1))
            if len(errs) > 1
            else 0.0
        ),
    }

    print(
        f"{condition_name:16s} "
        f"MSE={ADAPTIVE_RESULTS[condition_name]['mean_mse']:.6e} "
        f"+/- {ADAPTIVE_RESULTS[condition_name]['std_mse']:.2e} "
        f"obs={first['selected_obs']} "
        f"times={first['selected_times']}"
    )


plt.figure(figsize=(8, 4))
plt.plot(
    adaptive_history_task,
    label="task loss"
)
plt.plot(
    adaptive_history_total,
    label="total loss"
)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    "benchmark_adaptive_training_loss.png",
    dpi=160
)


# Hard adaptive policies across hardware conditions.
adaptive_hard_obs = torch.stack([
    ADAPTIVE_RESULTS[name]["hard_obs"]
    for name, _ in condition_items
])

adaptive_hard_time = torch.stack([
    ADAPTIVE_RESULTS[name]["hard_time"]
    for name, _ in condition_items
])

plt.figure(figsize=(11, 5))
plt.imshow(
    adaptive_hard_obs.numpy(),
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=1
)
plt.colorbar(label="hard selected")
plt.xticks(
    range(NOBS),
    observable_names,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(condition_items)),
    [name for name, _ in condition_items]
)
plt.tight_layout()
plt.savefig(
    "benchmark_adaptive_hard_obs.png",
    dpi=160
)

plt.figure(figsize=(8, 5))
plt.imshow(
    adaptive_hard_time.numpy(),
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=1
)
plt.colorbar(label="hard selected")
plt.xticks(
    range(NTIME),
    candidate_times
)
plt.yticks(
    range(len(condition_items)),
    [name for name, _ in condition_items]
)
plt.xlabel("reservoir step")
plt.tight_layout()
plt.savefig(
    "benchmark_adaptive_hard_time.png",
    dpi=160
)


# ------------------------------------------------------------
# 12. Benchmark fixed policies vs the jointly trained adaptive model
# ------------------------------------------------------------

condition_names = list(CONDITIONS.keys())

# Fixed benchmark columns + one adaptive column.
policy_names = (
    list(MEASUREMENT_POLICIES.keys())
    + ["adaptive_selector"]
)

mean_mse = np.zeros(
    (len(condition_names), len(policy_names))
)

std_mse = np.zeros_like(
    mean_mse
)

mean_best_epoch = np.full_like(
    mean_mse,
    np.nan
)

example_predictions = {}

print("\nTraining fixed-policy readers...")

for ic, condition_name in enumerate(condition_names):
    print(
        f"\n=== {condition_name} ==="
    )

    R_train_all = CACHE[condition_name]["train"]
    R_val_all = CACHE[condition_name]["val"]
    R_test_all = CACHE[condition_name]["test"]

    # --------------------------------------------------------
    # Fixed policies: same benchmark procedure as benchmark.py
    # --------------------------------------------------------
    for ip, policy_name in enumerate(
        MEASUREMENT_POLICIES.keys()
    ):
        obs_names = (
            MEASUREMENT_POLICIES[policy_name]
        )

        R_train = select_policy_features(
            R_train_all,
            obs_names
        )

        R_val = select_policy_features(
            R_val_all,
            obs_names
        )

        R_test = select_policy_features(
            R_test_all,
            obs_names
        )

        errors = []
        best_epochs = []

        for r in range(N_READOUT_SEEDS):
            model, val_loss, best_epoch = train_readout(
                R_train,
                train_u,
                R_val,
                val_u,
                seed=(
                    10000
                    + 1000 * ic
                    + 100 * ip
                    + r
                )
            )

            model.eval()

            with torch.no_grad():
                _, pred_test = model(
                    R_test
                )

                err = mse(
                    pred_test,
                    test_u
                ).item()

            errors.append(
                err
            )

            best_epochs.append(
                best_epoch
            )

            if r == 0:
                example_predictions[
                    (condition_name, policy_name)
                ] = pred_test.detach().cpu()

        mean_mse[ic, ip] = np.mean(
            errors
        )

        std_mse[ic, ip] = (
            np.std(errors, ddof=1)
            if len(errors) > 1
            else 0.0
        )

        mean_best_epoch[ic, ip] = np.mean(
            best_epochs
        )

        print(
            f"{policy_name:18s} "
            f"{str(obs_names):38s} "
            f"MSE={mean_mse[ic, ip]:.6e} "
            f"+/- {std_mse[ic, ip]:.2e} "
            f"epoch~{mean_best_epoch[ic, ip]:.0f}"
        )

    # --------------------------------------------------------
    # Adaptive model:
    # NO fresh readout training here.
    # Use the readout jointly trained with the controller.
    # --------------------------------------------------------
    ip = len(policy_names) - 1

    adaptive = ADAPTIVE_RESULTS[
        condition_name
    ]

    mean_mse[ic, ip] = adaptive[
        "mean_mse"
    ]

    std_mse[ic, ip] = adaptive[
        "std_mse"
    ]

    example_predictions[
        (condition_name, "adaptive_selector")
    ] = adaptive["u_hat"]

    print(
        f"{'adaptive_selector':18s} "
        f"obs={str(adaptive['selected_obs']):30s} "
        f"times={str(adaptive['selected_times']):18s} "
        f"MSE={mean_mse[ic, ip]:.6e} "
        f"+/- {std_mse[ic, ip]:.2e}"
    )

# ------------------------------------------------------------
# 13. Print rankings
# ------------------------------------------------------------

print("\n================ RANKINGS ================")

for ic, condition_name in enumerate(condition_names):
    order = np.argsort(mean_mse[ic])

    print(f"\n{condition_name}")

    for rank, ip in enumerate(order, start=1):
        print(
            f"{rank:2d}. {policy_names[ip]:14s} "
            f"MSE={mean_mse[ic, ip]:.6e}  "
            f"{({'obs': ADAPTIVE_RESULTS[condition_name]['selected_obs'], 'times': ADAPTIVE_RESULTS[condition_name]['selected_times']} if policy_names[ip] == 'adaptive_selector' else MEASUREMENT_POLICIES[policy_names[ip]])}"
        )

# ------------------------------------------------------------
# 14. Test-MSE heatmap
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

im = plt.imshow(
    mean_mse,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(im, label="test MSE")

plt.xticks(
    range(len(policy_names)),
    policy_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(condition_names)),
    condition_names
)

for i in range(len(condition_names)):
    for j in range(len(policy_names)):
        plt.text(
            j, i,
            f"{mean_mse[i, j]:.3f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()
plt.savefig("benchmark_mse_heatmap.png", dpi=160)

# ------------------------------------------------------------
# 15. Relative regret
#
# 1 means best tested policy for this condition.
# >1 tells how much worse the fixed choice is.
# ------------------------------------------------------------

best = mean_mse.min(axis=1, keepdims=True)
regret = mean_mse / best

plt.figure(figsize=(12, 5))

im = plt.imshow(
    regret,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(im, label="MSE / best MSE in condition")

plt.xticks(
    range(len(policy_names)),
    policy_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(condition_names)),
    condition_names
)

for i in range(len(condition_names)):
    for j in range(len(policy_names)):
        plt.text(
            j, i,
            f"{regret[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()
plt.savefig("benchmark_relative_regret.png", dpi=160)

# ------------------------------------------------------------
# 16. Grouped bars with initialization error bars
# ------------------------------------------------------------

x = np.arange(len(policy_names))
width = 0.12

plt.figure(figsize=(13, 5))

for ic, condition_name in enumerate(condition_names):
    dx = (ic - (len(condition_names) - 1) / 2) * width

    plt.bar(
        x + dx,
        mean_mse[ic],
        width=width,
        yerr=std_mse[ic],
        capsize=2,
        label=condition_name
    )

plt.xticks(
    x,
    policy_names,
    rotation=45,
    ha="right"
)

plt.ylabel("test MSE")
plt.legend()
plt.tight_layout()
plt.savefig("benchmark_mse_bars.png", dpi=160)

# ------------------------------------------------------------
# 17. Best-vs-worst reconstruction example
# ------------------------------------------------------------

idx_example = 3

fig, axes = plt.subplots(
    len(condition_names),
    1,
    figsize=(8, 2.7 * len(condition_names)),
    sharex=True
)

for ic, condition_name in enumerate(condition_names):
    best_ip = int(np.argmin(mean_mse[ic]))
    worst_ip = int(np.argmax(mean_mse[ic]))

    best_name = policy_names[best_ip]
    worst_name = policy_names[worst_ip]

    ax = axes[ic]

    ax.plot(
        times.cpu(),
        test_u[idx_example].cpu(),
        label="true"
    )

    ax.plot(
        times.cpu(),
        example_predictions[(condition_name, best_name)][idx_example],
        "--",
        label=f"best: {best_name}"
    )

    ax.plot(
        times.cpu(),
        example_predictions[(condition_name, worst_name)][idx_example],
        ":",
        label=f"worst: {worst_name}"
    )

    ax.set_title(condition_name)
    ax.set_ylabel("u(t)")
    ax.legend(fontsize=8)

axes[-1].set_xlabel("t")

plt.tight_layout()
plt.savefig("benchmark_best_vs_worst.png", dpi=160)

# ------------------------------------------------------------
# 18. Save numerical results
# ------------------------------------------------------------

np.savez(
    "benchmark_results.npz",
    mean_mse=mean_mse,
    std_mse=std_mse,
    mean_best_epoch=mean_best_epoch,
    condition_names=np.array(condition_names),
    policy_names=np.array(policy_names),
)

print("\nSaved:")
print("  benchmark_mse_heatmap.png")
print("  benchmark_relative_regret.png")
print("  benchmark_mse_bars.png")
print("  benchmark_best_vs_worst.png")
print("  benchmark_results.npz")
print("  benchmark_adaptive_training_loss.png")
print("  benchmark_adaptive_selector_scores.png")
