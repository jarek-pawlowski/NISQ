import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. 4-spin Heisenberg reservoir
# ============================================================

N = 4
DIM = 2**N

I = torch.eye(2, dtype=torch.complex128, device=device)
X = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128, device=device)
Y = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex128, device=device)
Z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128, device=device)


def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = torch.kron(out, op)
    return out


def op1(A, i):
    ops = [I] * N
    ops[i] = A
    return kron_all(ops)


def op2(A, i, B, j):
    ops = [I] * N
    ops[i] = A
    ops[j] = B
    return kron_all(ops)


J = [1.00, 0.83, 1.17]
h = [0.21, -0.13, 0.08, 0.17]

H = torch.zeros((DIM, DIM), dtype=torch.complex128, device=device)

for i in range(N - 1):
    H += J[i] * (
        op2(X, i, X, i+1)
        + op2(Y, i, Y, i+1)
        + op2(Z, i, Z, i+1)
    )  # *10.

for i in range(N):
    H += h[i] * op1(Z, i)

dt_res = 0.35
Ures = torch.matrix_exp(-1j * H * dt_res)
Y0 = op1(Y, 0)

# observables: local X,Y,Z + nearest-neighbour ZZ
observables = []

for i in range(N):
    observables += [op1(X, i), op1(Y, i), op1(Z, i)]

for i in range(N - 1):
    observables.append(op2(Z, i, Z, i+1))


# ============================================================
# 2. Reservoir dynamics
# ============================================================

def input_unitary(u, strength=0.8):
    return torch.matrix_exp(-0.5j * strength * u * Y0)


def zero_state():
    psi = torch.zeros(DIM, dtype=torch.complex128, device=device)
    psi[0] = 1.
    return psi


def expectation(psi, O):
    return torch.real(torch.vdot(psi, O @ psi))


@torch.no_grad()
def reservoir_features(signal, sample_every=5):
    psi = zero_state()
    features = []

    for n, u in enumerate(signal):
        psi = input_unitary(u) @ psi
        psi = Ures @ psi
        
        if n % sample_every == 0:
            f = torch.stack([
                expectation(psi, O)
                for O in observables
            ])
            features.append(f)

    return torch.stack(features).flatten()


# ============================================================
# 3. Dataset
# ============================================================

T = 50
times = torch.linspace(0, 2 * math.pi, T, device=device)


def generate_signals(n):
    A1 = 0.4 + 0.6 * torch.rand(n, device=device)
    A2 = 0.2 + 0.5 * torch.rand(n, device=device)

    w1 = 0.7 + 1.3 * torch.rand(n, device=device)
    w2 = 2.2 + 1.5 * torch.rand(n, device=device)

    p1 = 2 * math.pi * torch.rand(n, device=device)
    p2 = 2 * math.pi * torch.rand(n, device=device)

    t = times[None, :]

    u = (
        A1[:, None] * torch.sin(w1[:, None] * t + p1[:, None])
        + A2[:, None] * torch.sin(w2[:, None] * t + p2[:, None])
    )

    params = torch.stack([A1, w1, p1, A2, w2, p2], dim=1)
    return u, params


NTRAIN = 400  # 4000
NTEST = 100

train_u, train_params = generate_signals(NTRAIN)
test_u, test_params = generate_signals(NTEST)

print("Generating reservoir data...")

def make_R(signals):
    return torch.stack([reservoir_features(u, sample_every=5) for u in signals])

Rtrain = make_R(train_u)
Rtest = make_R(test_u)

NR = Rtrain.shape[1]
NFREQ = T // 2 + 1

print("reservoir dimension:", NR)


# ============================================================
# 4. Common MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, nout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# A: R -> FFT, supervised FFT labels
# ============================================================

class Model_supervised(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = MLP(NR, 2 * NFREQ)

    def forward(self, R):
        x = self.readout(R)
        return torch.complex(x[:, :NFREQ], x[:, NFREQ:])


# ============================================================
# B: R -> Fourier coefficients -> IFFT (physics decoder)
# ============================================================

class Model_pinn(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = MLP(NR, 2 * NFREQ)

    def forward(self, R):
        x = self.readout(R)
        F = torch.complex(x[:, :NFREQ], x[:, NFREQ:])
        u = torch.fft.irfft(F, n=T, dim=1)
        return F, u

modelS = Model_supervised().to(device)
modelP = Model_pinn().to(device)

# ============================================================
# 5. Training
# ============================================================

Ftrain = torch.fft.rfft(train_u, dim=1)

mse = nn.MSELoss()

def complex_mse(a, b):
    return mse(a.real, b.real) + mse(a.imag, b.imag)

optS = torch.optim.Adam(modelS.parameters(), lr=2e-3, weight_decay=1e-4)
optP = torch.optim.Adam(modelP.parameters(), lr=2e-3, weight_decay=1e-4)

EPOCHS = 5000
historyS, historyP = [], []


for epoch in range(EPOCHS):

    # A: supervised FFT
    optS.zero_grad()
    FS = modelS(Rtrain)
    lossS = complex_mse(FS, Ftrain)
    lossS.backward()
    optS.step()

    # B: reconstruction through fixed IFFT
    optP.zero_grad()
    FP, uP = modelP(Rtrain)
    lossP = mse(uP, train_u)
    lossP.backward()
    optP.step()

    # compare both in signal space
    with torch.no_grad():
        FS_test = modelS(Rtest)
        uS_test = torch.fft.irfft(FS_test, n=T, dim=1)
        _, uP_test = modelP(Rtest)

        errS = mse(uS_test, test_u).item()
        errP = mse(uP_test, test_u).item()

    historyS.append(errS)
    historyP.append(errP)

    if epoch % 100 == 0:
        print(
            f"{epoch:4d} | "
            f"S={errS:.6e} | "
            f"P={errP:.6e}"
        )
        

# ============================================================
# 6. Final comparison
# ============================================================

modelS.eval()
modelP.eval()

with torch.no_grad():
    FS = modelS(Rtest)
    predS = torch.fft.irfft(FS, n=T, dim=1)
    FP, predP = modelP(Rtest)

print("\nFINAL TEST MSE")
print("S:", mse(predS, test_u).item())
print("P:", mse(predP, test_u).item())


# ============================================================
# 7. Plot example
# ============================================================

idx = 5

plt.figure(figsize=(9, 5))
plt.plot(times.cpu(), test_u[idx].cpu(), label="true", linewidth=3)
plt.plot(times.cpu(), predS[idx].cpu(), "--", label="S: supervised FFT")
plt.plot(times.cpu(), predP[idx].cpu(), "--", label="P: physics-aware Fourier decoder")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png")


# learning curves
plt.figure(figsize=(8, 5))
plt.semilogy(historyS, label="S")
plt.semilogy(historyP, label="P")
plt.xlabel("epoch")
plt.ylabel("test signal MSE")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curves.png")
