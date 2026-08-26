import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# Adaptive physics-informed quantum reservoir computing -- v2
# ============================================================

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
torch.manual_seed(7)

# ------------------------------------------------------------
# 1. Basic operators
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
    # Extra measurement-noise stress test is applied only to local
    # X/Y/Z observables, not to ZZ correlations.
    observable_qubit.append(None)
NOBS = len(observables)
print("number of candidate observables:", NOBS)

# ------------------------------------------------------------
# 2. Dataset
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
    u = (
        A1[:, None] * torch.sin(w1[:, None] * t + p1[:, None])
        + A2[:, None] * torch.sin(w2[:, None] * t + p2[:, None])
    )
    params = torch.stack([A1, w1, p1, A2, w2, p2], dim=1)
    return u, params


NTRAIN = 1000
NTEST = 250
train_u, _ = generate_signals(NTRAIN)
test_u, _ = generate_signals(NTEST)

# ------------------------------------------------------------
# 3. Candidate measurement grid + fixed budget
# ------------------------------------------------------------
SAMPLE_EVERY = 5
candidate_times = list(range(0, T, SAMPLE_EVERY))
NTIME = len(candidate_times)
NR = NTIME * NOBS
READOUT_DIM = NR + NOBS + NTIME  # 150 + 15 + 10 = 175

K_OBS = 4
K_TIME = 2 # 4

print("candidate measurement times:", candidate_times)
print("maximum reservoir feature dimension:", NR)
print(f"hard measurement budget: {K_OBS} observables x {K_TIME} times = {K_OBS*K_TIME}")

# ------------------------------------------------------------
# 4. Hardware/noise conditions + controller context
# ------------------------------------------------------------

GAMMA_PHI_MAX = 0.30
GAMMA_1_MAX = 0.12
MEAS_NOISE_MAX = 0.35

# task id + gamma_phi[4] + gamma_1[4] + measurement_noise[4] + nshots
CONTEXT_DIM = 1 + N + N + N + 1


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

    # Strongly heterogeneous extrapolation case.
    "noisy": {
        "gamma_phi": [0.25, 0.22, 0.30, 0.20],
        "gamma_1":   [0.10, 0.08, 0.12, 0.07],
        "nshots": 300,
        "measurement_noise_qubit": None,
        "measurement_noise_std": 0.0,
    },
}


def measurement_noise_vector(measurement_noise_qubit, measurement_noise_std):
    """
    Return per-qubit measurement-noise amplitudes [4].

    Example:
        bad_q0_meas -> [0.35, 0, 0, 0]
        bad_q2_meas -> [0, 0, 0.35, 0]
    """
    v = torch.zeros(N, device=device)

    if measurement_noise_qubit is not None:
        v[int(measurement_noise_qubit)] = float(measurement_noise_std)

    return v


def controller_context(
    task_id,
    gamma_phi,
    gamma_1,
    nshots,
    measurement_noise_qubit=None,
    measurement_noise_std=0.0,
):
    """
    Extended hardware context seen by AuxiliaryController:

        [ task_id,
          gamma_phi_0 ... gamma_phi_3,
          gamma_1_0   ... gamma_1_3,
          sigma_meas_0 ... sigma_meas_3,
          log10(nshots) ]

    Total dimension = 1 + 4 + 4 + 4 + 1 = 14.
    """
    gamma_phi = torch.as_tensor(
        gamma_phi, dtype=torch.get_default_dtype(), device=device
    )
    gamma_1 = torch.as_tensor(
        gamma_1, dtype=torch.get_default_dtype(), device=device
    )
    nshots = torch.as_tensor(
        float(nshots), dtype=torch.get_default_dtype(), device=device
    )

    gp = gamma_phi / GAMMA_PHI_MAX
    g1 = gamma_1 / GAMMA_1_MAX

    gm = measurement_noise_vector(
        measurement_noise_qubit,
        measurement_noise_std
    ) / MEAS_NOISE_MAX

    ns = (torch.log10(nshots) - 2.0) / 2.0

    task = torch.as_tensor(
        [task_id],
        device=device,
        dtype=torch.get_default_dtype()
    )

    return torch.cat([
        task,
        gp,
        g1,
        gm,
        ns.reshape(1),
    ])[None, :]


def condition_to_tensors(cfg):
    gamma_phi = torch.tensor(
        cfg["gamma_phi"], dtype=torch.get_default_dtype(), device=device
    )
    gamma_1 = torch.tensor(
        cfg["gamma_1"], dtype=torch.get_default_dtype(), device=device
    )
    nshots = torch.tensor(
        float(cfg["nshots"]), dtype=torch.get_default_dtype(), device=device
    )

    return gamma_phi, gamma_1, nshots

# ------------------------------------------------------------
# 5. Auxiliary controller
# ------------------------------------------------------------
class AuxiliaryController(nn.Module):
    def __init__(self, hidden=48, policy_beta=4.0):
        super().__init__()
        self.policy_beta = policy_beta
        self.backbone = nn.Sequential(
            nn.Linear(CONTEXT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
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
        raw = self.head_physics(z)

        if TRAIN_RESERVOIR_CONTROLS:
            J = (0.5 + 1.0 * torch.sigmoid(raw[..., 0:3])).squeeze(0)
            h = (0.4 * torch.tanh(raw[..., 3:7])).squeeze(0)
            dt = (0.15 + 0.45 * torch.sigmoid(raw[..., 7:8])).squeeze()
            alpha = (0.4 + 0.8 * torch.sigmoid(raw[..., 8:9])).squeeze()
        else:
            J = J_FIXED
            h = H_FIXED
            dt = DT_FIXED
            alpha = ALPHA_FIXED

        delta_obs = self.obs_correction(z).squeeze(0)
        delta_time = self.time_correction(z).squeeze(0)

        logits_obs = self.base_obs + self.policy_beta * delta_obs
        logits_time = self.base_time + self.policy_beta * delta_time

        return {
            "J": J,
            "h": h,
            "dt": dt,
            "alpha": alpha,
            "logits_obs": logits_obs,
            "logits_time": logits_time,
            "base_obs": self.base_obs,
            "base_time": self.base_time,
            "delta_obs": delta_obs,
            "delta_time": delta_time,
        }

# ------------------------------------------------------------
# 6. Competitive selector: softmax warm-up + Gumbel-TopK
# ------------------------------------------------------------
def allocation_scores(logits, tau=1.0):
    return torch.softmax(logits / tau, dim=-1)


def soft_budget_gate(logits, k, tau=1.0):
    # Competitive soft allocation; sums exactly to k.
    soft = k * torch.softmax(logits / tau, dim=-1)
    return soft, torch.softmax(logits / tau, dim=-1)


def straight_through_topk(logits, k, tau=1.0, training=True):
    # Gumbel-TopK during training, deterministic TopK at evaluation.
    if training:
        eps = 1e-10
        u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        gumbel = -torch.log(-torch.log(u))
        scores = logits + gumbel
    else:
        scores = logits

    soft = k * torch.softmax(scores / tau, dim=-1)
    idx = torch.topk(scores, k=k, dim=-1).indices
    hard = torch.zeros_like(logits)
    hard[idx] = 1.0

    if training:
        gate = hard.detach() - soft.detach() + soft
    else:
        gate = hard

    return gate, torch.softmax(logits / tau, dim=-1), hard

# ------------------------------------------------------------
# 7. Reservoir Hamiltonian + local heterogeneous noise
# ------------------------------------------------------------
def build_H(theta):
    H = torch.zeros((DIM, DIM), dtype=torch.complex128, device=device)
    for i in range(N - 1):
        H = H + theta["J"][i] * (XX[i] + YY[i] + ZZ[i])
    for i in range(N):
        H = H + theta["h"][i] * Zq[i]
    return H


def embed_single_qubit_matrix(A, site):
    ops = [I2 for _ in range(N)]
    ops[site] = A
    return kron_all(ops)


def make_noise_operators(gamma_phi, gamma_1, dt):
    p_phi = 0.5 * (1.0 - torch.exp(-2.0 * gamma_phi * dt))
    p_1 = 1.0 - torch.exp(-gamma_1 * dt)
    p_phi = torch.clamp(p_phi, 0.0, 0.499999)
    p_1 = torch.clamp(p_1, 0.0, 0.999999)

    K0, K1 = [], []
    for i in range(N):
        pi = p_1[i]
        K0_1q = torch.stack([
            torch.stack([
                torch.ones((), device=device, dtype=torch.complex128),
                torch.zeros((), device=device, dtype=torch.complex128),
            ]),
            torch.stack([
                torch.zeros((), device=device, dtype=torch.complex128),
                torch.sqrt(1.0 - pi).to(torch.complex128),
            ])
        ])
        K1_1q = torch.stack([
            torch.stack([
                torch.zeros((), device=device, dtype=torch.complex128),
                torch.sqrt(pi).to(torch.complex128),
            ]),
            torch.stack([
                torch.zeros((), device=device, dtype=torch.complex128),
                torch.zeros((), device=device, dtype=torch.complex128),
            ])
        ])
        K0.append(embed_single_qubit_matrix(K0_1q, i))
        K1.append(embed_single_qubit_matrix(K1_1q, i))
    return p_phi, K0, K1


def apply_noise(rho, p_phi, K0, K1):
    for i in range(N):
        Z_i = Zq[i]
        zrho = Z_i @ rho @ Z_i
        rho = (1.0 - p_phi[i]) * rho + p_phi[i] * zrho
        k0 = K0[i]
        k1 = K1[i]
        rho = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
    return rho

# ------------------------------------------------------------
# 8. Batched differentiable reservoir
# ------------------------------------------------------------
def initial_density_matrix(batch_size):
    psi0 = torch.zeros(DIM, dtype=torch.complex128, device=device)
    psi0[0] = 1.0
    rho0 = torch.outer(psi0, psi0.conj())
    return rho0.unsqueeze(0).repeat(batch_size, 1, 1)


def expectation_batch(rho, O):
    return torch.real(torch.einsum("bij,ji->b", rho, O))


def reservoir_forward(
    signals,
    theta,
    gamma_phi,
    gamma_1,
    nshots,
    gate_obs,
    gate_time,
    add_shot_noise=True,
    measurement_noise_qubit=None,
    measurement_noise_std=0.0,
):
    B = signals.shape[0]
    H = build_H(theta)
    Ures = torch.matrix_exp(-1j * H * theta["dt"])

    p_phi, K0, K1 = make_noise_operators(gamma_phi, gamma_1, theta["dt"])

    rho = initial_density_matrix(B)
    features = []
    time_slot = 0

    for n in range(T):
        angle = theta["alpha"] * signals[:, n]
        c = torch.cos(0.5 * angle)[:, None, None].to(torch.complex128)
        s = torch.sin(0.5 * angle)[:, None, None].to(torch.complex128)
        Uin = c * I16[None, :, :] - 1j * s * Y0[None, :, :]

        rho = Uin @ rho @ Uin.conj().transpose(-1, -2)
        rho = Ures @ rho @ Ures.conj().T
        rho = apply_noise(rho, p_phi, K0, K1)

        if n % SAMPLE_EVERY == 0:
            vals = torch.stack([expectation_batch(rho, O) for O in observables], dim=1)

            if add_shot_noise:
                var = torch.clamp(1.0 - vals**2, min=1e-10)
                std = torch.sqrt(var / nshots)
                vals = vals + torch.randn_like(vals) * std

            # Extra qubit-local readout noise.  It acts AFTER the quantum
            # evolution and BEFORE the observable gate is applied.
            # Thus avoiding a corrupted observable can genuinely help.
            if (
                measurement_noise_qubit is not None
                and measurement_noise_std > 0.0
            ):
                for k, q in enumerate(observable_qubit):
                    if q == int(measurement_noise_qubit):
                        vals[:, k] = (
                            vals[:, k]
                            + float(measurement_noise_std)
                            * torch.randn_like(vals[:, k])
                        )

            vals = vals * gate_obs[None, :]
            vals = vals * gate_time[time_slot]
            features.append(vals)
            time_slot += 1

    return torch.stack(features, dim=1).flatten(start_dim=1)

# ------------------------------------------------------------
# 9. Physics-informed Fourier readout
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
        F = torch.complex(x[:, :NFREQ], x[:, NFREQ:])
        u = torch.fft.irfft(F, n=T, dim=1)
        return F, u


# First run: isolate adaptive sensing.
# Set True only after confirming that the measurement policy adapts.
TRAIN_RESERVOIR_CONTROLS = False

J_FIXED = torch.tensor([1.00, 0.83, 1.17], device=device)
H_FIXED = torch.tensor([0.21, -0.13, 0.08, 0.17], device=device)
DT_FIXED = torch.tensor(0.35, device=device)
ALPHA_FIXED = torch.tensor(0.80, device=device)



def build_readout_input(R, gate_obs, gate_time):
    """
    Concatenate reservoir features with explicit measurement metadata.

    Inputs:
        R         : [B, NR]
        gate_obs  : [NOBS]
        gate_time : [NTIME]

    Output:
        [B, NR + NOBS + NTIME]

    For the present setup: 150 + 15 + 10 = 175.
    """
    B = R.shape[0]
    obs_meta = gate_obs.unsqueeze(0).expand(B, -1)
    time_meta = gate_time.unsqueeze(0).expand(B, -1)
    return torch.cat([R, obs_meta, time_meta], dim=1)


controller = AuxiliaryController(hidden=64, policy_beta=4.0).to(device)
readout = PhysicsReadout(READOUT_DIM).to(device)

# ------------------------------------------------------------
# 10. Training configuration
# ------------------------------------------------------------
N_DEVICE_BATCH = len(CONDITIONS)  # one sub-batch per explicit hardware condition
SIGNALS_PER_DEVICE = 8
EPOCHS = 4000
# Keep warm-up lengths fixed when extending training.
SOFT_WARMUP_EPOCHS = 600
ENTROPY_WARMUP_EPOCHS = 400
TAU_START = 2.0
TAU_END = 0.75
TAU_ANNEAL_EPOCHS = 2000
LAMBDA_CTRL = 1e-4
LAMBDA_ENTROPY = 2e-3

optimizer = torch.optim.AdamW(
    list(controller.parameters()) + list(readout.parameters()),
    lr=2e-3,
    weight_decay=1e-4,
)
# The loss is still decreasing at ~2000 epochs; continue training,
# but reduce LR after the coarse policy has formed.
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[1800, 3000], gamma=0.3
)
mse = nn.MSELoss()

history_task = []
history_total = []
history_entropy = []
history_policy_delta = []

# ------------------------------------------------------------
# 11. Training
# ------------------------------------------------------------

condition_items = list(CONDITIONS.items())

for epoch in range(EPOCHS):
    controller.train()
    readout.train()

    tau_frac = min(epoch / max(TAU_ANNEAL_EPOCHS - 1, 1), 1.0)
    tau = TAU_START * (TAU_END / TAU_START) ** tau_frac

    soft_phase = epoch < SOFT_WARMUP_EPOCHS
    entropy_phase = epoch < ENTROPY_WARMUP_EPOCHS

    total_task = 0.0
    total_ctrl = 0.0
    total_entropy = 0.0
    total_delta_norm = 0.0

    # IMPORTANT:
    # every optimizer step sees every hardware condition.
    # This directly trains one conditional policy
    #
    #       hardware context -> measurement policy.
    #
    for condition_name, cfg in condition_items:
        gamma_phi, gamma_1, nshots = condition_to_tensors(cfg)

        context = controller_context(
            1.0,
            gamma_phi,
            gamma_1,
            nshots,
            measurement_noise_qubit=cfg["measurement_noise_qubit"],
            measurement_noise_std=cfg["measurement_noise_std"],
        )

        theta = controller(context)

        if soft_phase:
            gate_obs, _ = soft_budget_gate(
                theta["logits_obs"], K_OBS, tau=tau
            )
            gate_time, _ = soft_budget_gate(
                theta["logits_time"], K_TIME, tau=tau
            )
        else:
            gate_obs, _, _ = straight_through_topk(
                theta["logits_obs"],
                K_OBS,
                tau=tau,
                training=True
            )
            gate_time, _, _ = straight_through_topk(
                theta["logits_time"],
                K_TIME,
                tau=tau,
                training=True
            )

        ids = torch.randint(
            0,
            NTRAIN,
            (SIGNALS_PER_DEVICE,),
            device=device
        )
        u = train_u[ids]

        R = reservoir_forward(
            signals=u,
            theta=theta,
            gamma_phi=gamma_phi,
            gamma_1=gamma_1,
            nshots=nshots,
            gate_obs=gate_obs,
            gate_time=gate_time,
            add_shot_noise=True,
            measurement_noise_qubit=cfg["measurement_noise_qubit"],
            measurement_noise_std=cfg["measurement_noise_std"],
        )

        readout_input = build_readout_input(
            R,
            gate_obs,
            gate_time
        )

        _, u_hat = readout(readout_input)
        task_loss = mse(u_hat, u)

        q_obs = allocation_scores(
            theta["logits_obs"], tau=1.0
        )
        q_time = allocation_scores(
            theta["logits_time"], tau=1.0
        )

        entropy = (
            -(q_obs * torch.log(q_obs + 1e-12)).sum()
            -(q_time * torch.log(q_time + 1e-12)).sum()
        )
        entropy_loss = -entropy

        if TRAIN_RESERVOIR_CONTROLS:
            ctrl_loss = (
                ((theta["J"] - 1.0)**2).mean()
                + (theta["h"]**2).mean()
                + (theta["dt"] - 0.35)**2
                + (theta["alpha"] - 0.8)**2
            )
        else:
            ctrl_loss = torch.zeros((), device=device)

        delta_norm = (
            theta["delta_obs"].pow(2).mean()
            + theta["delta_time"].pow(2).mean()
        )

        total_task += task_loss
        total_ctrl += ctrl_loss
        total_entropy += entropy_loss
        total_delta_norm += delta_norm

    n_conditions = len(condition_items)

    total_task /= n_conditions
    total_ctrl /= n_conditions
    total_entropy /= n_conditions
    total_delta_norm /= n_conditions

    loss = total_task + LAMBDA_CTRL * total_ctrl

    if entropy_phase:
        loss = loss + LAMBDA_ENTROPY * total_entropy

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(controller.parameters()) + list(readout.parameters()),
        max_norm=5.0
    )

    optimizer.step()
    scheduler.step()

    history_task.append(total_task.item())
    history_total.append(loss.item())
    history_entropy.append(total_entropy.item())
    history_policy_delta.append(total_delta_norm.item())

    if epoch % 50 == 0:
        phase_name = "soft" if soft_phase else "gumbel-topk"

        print(
            f"{epoch:4d} | "
            f"phase={phase_name:11s} | "
            f"task={total_task.item():.5e} | "
            f"delta={total_delta_norm.item():.3e} | "
            f"tau={tau:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

# ------------------------------------------------------------
# 12. Evaluation helper
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_condition(cfg, ntest=100):
    controller.eval()
    readout.eval()

    gamma_phi, gamma_1, nshots = condition_to_tensors(cfg)

    context = controller_context(
        1.0,
        gamma_phi,
        gamma_1,
        nshots,
        measurement_noise_qubit=cfg["measurement_noise_qubit"],
        measurement_noise_std=cfg["measurement_noise_std"],
    )

    theta = controller(context)

    _, prob_obs, hard_obs = straight_through_topk(
        theta["logits_obs"],
        K_OBS,
        tau=TAU_END,
        training=False
    )

    _, prob_time, hard_time = straight_through_topk(
        theta["logits_time"],
        K_TIME,
        tau=TAU_END,
        training=False
    )

    u = test_u[:ntest]

    R = reservoir_forward(
        signals=u,
        theta=theta,
        gamma_phi=gamma_phi,
        gamma_1=gamma_1,
        nshots=nshots,
        gate_obs=hard_obs,
        gate_time=hard_time,
        add_shot_noise=True,
        measurement_noise_qubit=cfg["measurement_noise_qubit"],
        measurement_noise_std=cfg["measurement_noise_std"],
    )

    readout_input = build_readout_input(
        R,
        hard_obs,
        hard_time
    )

    _, u_hat = readout(readout_input)
    err = mse(u_hat, u).item()

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
        "theta": theta,
        "selected_obs": selected_obs,
        "selected_times": selected_times,
        "prob_obs": prob_obs.cpu(),
        "prob_time": prob_time.cpu(),
        "hard_obs": hard_obs.cpu(),
        "hard_time": hard_time.cpu(),
        "delta_obs": theta["delta_obs"].cpu(),
        "delta_time": theta["delta_time"].cpu(),
        "u": u.cpu(),
        "u_hat": u_hat.cpu(),
    }

# ------------------------------------------------------------
# 13. Representative hardware conditions
# ------------------------------------------------------------

results = []

for name, cfg in CONDITIONS.items():
    out = evaluate_condition(cfg)
    results.append((name, out))

    theta = out["theta"]

    print("\n================================================")
    print("condition:", name)
    print("gamma_phi:", cfg["gamma_phi"])
    print("gamma_1:", cfg["gamma_1"])
    print("shots:", cfg["nshots"])
    print(
        "measurement noise:",
        cfg["measurement_noise_qubit"],
        cfg["measurement_noise_std"]
    )
    print("test MSE:", out["mse"])

    if TRAIN_RESERVOIR_CONTROLS:
        print("J:", theta["J"].detach().cpu().numpy())
        print("h:", theta["h"].detach().cpu().numpy())
        print("dt:", theta["dt"].item())
        print("alpha:", theta["alpha"].item())

    print("selected observables:", out["selected_obs"])
    print("selected times:", out["selected_times"])

# ------------------------------------------------------------
# 14. Diagnostics / plots
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.semilogy(history_task, label="task loss")
plt.semilogy(history_total, label="total loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig("adaptive_v5_losses.png", dpi=150)

plt.figure(figsize=(8, 4))
plt.plot(history_policy_delta)
plt.xlabel("epoch")
plt.ylabel("noise-dependent policy correction norm")
plt.tight_layout()
plt.savefig("adaptive_v5_policy_delta.png", dpi=150)

for name, out in results:
    plt.figure(figsize=(9, 4))
    plt.bar(range(NOBS), out["prob_obs"].numpy())
    plt.xticks(range(NOBS), observable_names, rotation=45)
    plt.ylabel("allocation score")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"adaptive_v5_obs_{name}.png", dpi=150)

    plt.figure(figsize=(8, 4))
    plt.bar(range(NTIME), out["prob_time"].numpy())
    plt.xticks(range(NTIME), candidate_times)
    plt.xlabel("reservoir step")
    plt.ylabel("allocation score")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"adaptive_v5_time_{name}.png", dpi=150)

# ------------------------------------------------------------
# 15. Compact selector diagnostics across the six conditions
# ------------------------------------------------------------

# Hard observable and time policies.
hard_obs_rows = torch.stack(
    [out["hard_obs"] for _, out in results]
)
hard_time_rows = torch.stack(
    [out["hard_time"] for _, out in results]
)

plt.figure(figsize=(11, 5))
plt.imshow(
    hard_obs_rows.numpy(),
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=1
)
plt.colorbar(label="hard Top-4 selected")
plt.xticks(
    range(NOBS),
    observable_names,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(results)),
    [name for name, _ in results]
)
plt.tight_layout()
plt.savefig(
    "adaptive_v5_conditions_hard_obs.png",
    dpi=150
)

plt.figure(figsize=(9, 5))
plt.imshow(
    hard_time_rows.numpy(),
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=1
)
plt.colorbar(label="hard Top-4 selected")
plt.xticks(
    range(NTIME),
    candidate_times
)
plt.yticks(
    range(len(results)),
    [name for name, _ in results]
)
plt.xlabel("reservoir step")
plt.tight_layout()
plt.savefig(
    "adaptive_v5_conditions_hard_time.png",
    dpi=150
)


# Soft allocation scores are useful for seeing near-switches that do not
# yet cross the hard Top-K boundary.
soft_obs_rows = torch.stack(
    [out["prob_obs"] for _, out in results]
)
soft_time_rows = torch.stack(
    [out["prob_time"] for _, out in results]
)

plt.figure(figsize=(11, 5))
plt.imshow(
    soft_obs_rows.numpy(),
    aspect="auto",
    interpolation="nearest"
)
plt.colorbar(label="observable allocation score")
plt.xticks(
    range(NOBS),
    observable_names,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(results)),
    [name for name, _ in results]
)
plt.tight_layout()
plt.savefig(
    "adaptive_v5_conditions_soft_obs.png",
    dpi=150
)

plt.figure(figsize=(9, 5))
plt.imshow(
    soft_time_rows.numpy(),
    aspect="auto",
    interpolation="nearest"
)
plt.colorbar(label="time allocation score")
plt.xticks(
    range(NTIME),
    candidate_times
)
plt.yticks(
    range(len(results)),
    [name for name, _ in results]
)
plt.xlabel("reservoir step")
plt.tight_layout()
plt.savefig(
    "adaptive_v5_conditions_soft_time.png",
    dpi=150
)


# ------------------------------------------------------------
# 16. Optional continuous measurement-noise sweep
# ------------------------------------------------------------

@torch.no_grad()
def measurement_noise_sweep(
    qubit,
    npoints=31,
    base_gamma_phi=0.02,
    base_gamma_1=0.005,
    nshots=1000,
):
    """
    Sweep only the local measurement-noise amplitude sigma_meas,q.

    This directly checks whether the controller learns to reallocate
    observables as one detector becomes worse.
    """
    controller.eval()

    sweep = torch.linspace(
        0.0,
        MEAS_NOISE_MAX,
        npoints
    )

    obs_scores = []
    hard_obs = []

    for s in sweep:
        gp = [base_gamma_phi] * N
        g1 = [base_gamma_1] * N

        context = controller_context(
            1.0,
            gp,
            g1,
            nshots,
            measurement_noise_qubit=qubit,
            measurement_noise_std=float(s),
        )

        theta = controller(context)

        obs_scores.append(
            allocation_scores(
                theta["logits_obs"],
                tau=1.0
            ).cpu()
        )

        _, _, hard = straight_through_topk(
            theta["logits_obs"],
            K_OBS,
            tau=TAU_END,
            training=False
        )
        hard_obs.append(hard.cpu())

    return sweep, torch.stack(obs_scores), torch.stack(hard_obs)


for q in [0, 2]:
    sweep_meas, score_meas, hard_meas = measurement_noise_sweep(q)

    plt.figure(figsize=(9, 5))
    for k in range(NOBS):
        plt.plot(
            sweep_meas.numpy(),
            score_meas[:, k].numpy(),
            label=observable_names[k]
        )

    plt.xlabel(rf"$\sigma_{{\rm meas,{q}}}$")
    plt.ylabel("observable allocation score")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        f"adaptive_v5_meas_noise_sweep_q{q}.png",
        dpi=150
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(
        hard_meas.T.numpy(),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[
            0.0,
            MEAS_NOISE_MAX,
            -0.5,
            NOBS - 0.5
        ],
        vmin=0,
        vmax=1
    )
    plt.colorbar(label="hard Top-4 selected")
    plt.yticks(
        range(NOBS),
        observable_names
    )
    plt.xlabel(rf"$\sigma_{{\rm meas,{q}}}$")
    plt.ylabel("observable")
    plt.tight_layout()
    plt.savefig(
        f"adaptive_v5_meas_noise_hard_q{q}.png",
        dpi=150
    )


# ------------------------------------------------------------
# 17. Example reconstruction
# ------------------------------------------------------------

name, out = results[-1]
idx = 3

plt.figure(figsize=(8, 4))
plt.plot(
    times.cpu(),
    out["u"][idx],
    label="true"
)
plt.plot(
    times.cpu(),
    out["u_hat"][idx],
    "--",
    label="adaptive PI-QRC"
)
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title(name)
plt.legend()
plt.tight_layout()
plt.savefig(
    "adaptive_v5_conditions_example.png",
    dpi=150
)


print("\nHard policies:")
for name, out in results:
    print(
        f"{name:16s} "
        f"obs={out['selected_obs']} "
        f"times={out['selected_times']}"
    )

print("\nSaved diagnostic plots:")
print("  adaptive_v5_losses.png")
print("  adaptive_v5_policy_delta.png")
print("  adaptive_v5_obs_*.png")
print("  adaptive_v5_time_*.png")
print("  adaptive_v5_conditions_hard_obs.png")
print("  adaptive_v5_conditions_hard_time.png")
print("  adaptive_v5_conditions_soft_obs.png")
print("  adaptive_v5_conditions_soft_time.png")
print("  adaptive_v5_meas_noise_sweep_q0.png")
print("  adaptive_v5_meas_noise_sweep_q2.png")
print("  adaptive_v5_meas_noise_hard_q0.png")
print("  adaptive_v5_meas_noise_hard_q2.png")
print("  adaptive_v5_conditions_example.png")
