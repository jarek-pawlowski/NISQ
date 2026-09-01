# N=6 learned strong-to-weak (SW-like) witness

## Idea

We study a six-qubit open quantum system with a \(U(1)\) charge

\[
Q=\sum_i n_i,\qquad n_i=(1-Z_i)/2,
\]

and prepare the initial state in the fixed sector \(Q=3\).

The Hamiltonian conserves \(Q\), while the Lindblad generator contains both
charge-preserving channels (dephasing and incoherent hopping) and
charge-changing gain/loss channels. The latter allow the state to populate
several charge sectors. This is necessary for a strong-to-weak symmetry
crossover: a state that remains exactly inside one \(Q\) sector would remain
strong-symmetric.

For a density matrix \(\rho\) we calculate two offline diagnostics:

\[
C_{\rm lin}=|\mathrm{Tr}(\rho A_{1N})|
\]

and

\[
R_2=
\frac{\mathrm{Re}\,\mathrm{Tr}(\rho A_{1N}\rho A_{1N}^\dagger)}
{\mathrm{Tr}(\rho^2)},
\qquad
A_{1N}=S_1^+S_N^-.
\]

The finite-size SW-like regime is defined by first conditioning on

\[
C_{\rm lin}<\epsilon_C=0.02
\]

and then using

\[
y=1 \iff R_2>\epsilon_{R_2}=0.05.
\]

The Rényi correlator is used only to construct labels. It is **not** an input to
the neural network.

The learned witness sees only experimentally accessible one- and two-body
Pauli expectation values. There are 153 candidate observables. A trainable hard
Top-\(K\) selector chooses \(K=6\) of them and an MLP maps those measurements to
\(P_{\rm SW}\).

## Why continuous charge matching?

Gain/loss necessarily mixes charge sectors, but we do not want the network to
solve the task merely by measuring the amount of sector mixing.

Therefore, after imposing \(C_{\rm lin}<0.02\), states from the two \(R_2\)
classes are paired by nearest-neighbour matching in

\[
(p_{\rm out},\,\mathrm{Var}(Q),\,\langle Q\rangle).
\]

Each pair has almost identical simple charge diagnostics, while one state has
\(R_2\le0.05\) and the other has \(R_2>0.05\). Pair members must also originate
from different trajectories.

Importantly, trajectories are split into train/validation/test **before**
matching. Matching is then performed independently inside each split.

## Files

### `swssb_train.py`

This is the readable experiment pipeline. `main()` contains only the major
scientific steps:

1. `u.generate_raw_dataset(...)`
   - generates the restricted family of six-qubit Lindblad trajectories;

2. `u.prepare_matched_dataset(...)`
   - trajectory split;
   - condition \(C_{\rm lin}<0.02\);
   - continuous matching in charge diagnostics;

3. `u.normalize_matched_features(...)`
   - normalization from training data only;

4. `u.train_model(...)`
   - hard Top-\(K\) selector + MLP reader;

5. `u.evaluate_model(...)` and `u.charge_baselines(...)`
   - frozen test witness and simple charge-only controls;

6. output helpers
   - checkpoint, dataset, summary, ROC, training curves, selected measurements;

7. `u.run_final_demo(...)`
   - time-resolved sanity check for one fixed initial state.

### `utils.py`

Contains implementation details:

- **Physics**
  - operators and charge sector;
  - `build_hamiltonian`;
  - `liouvillian`;
  - time evolution;
  - \(C_{\rm lin}\), \(R_2\), and charge diagnostics.

- **Dataset**
  - targeted Lindblad parameter sampling;
  - trajectory generation;
  - 153 local Pauli features.

- **Matching**
  - `condition_on_low_clin`;
  - `split_trajectories_raw`;
  - `greedy_charge_match`;
  - `prepare_matched_dataset`.

- **Machine learning**
  - `HardTopKSelector`;
  - `SWWitness`;
  - training and evaluation.

- **Diagnostics / I/O**
  - ROC and training plots;
  - charge-only AUC comparison;
  - checkpoints and summaries;
  - final idealized/realistic trajectory demonstration.

## Final fixed-state test

At the end of every training run the code uses the same selected initial basis
state, by default

\[
|101010\rangle,
\]

which belongs to \(Q=3\).

Two evolutions are compared:

### Idealized

A clean homogeneous Hamiltonian with fixed dissipative rates. Gain/loss are
still present because sector mixing is required for the SW-like crossover.

### Realistic

The same initial state is evolved with one fixed disordered set of Hamiltonian
parameters and Lindblad rates drawn from the same parameter family used during
training.

For both cases the code saves

- `demo_idealized.png`
- `demo_realistic.png`

showing versus time:

1. \(C_{\rm lin}(t)\);
2. \(R_2(t)\);
3. learned \(P_{\rm SW}(t)\).

It also saves CSV files with \(C_{\rm lin}\), \(R_2\), witness probability,
\(\langle Q\rangle\), \(\mathrm{Var}(Q)\), and \(p_{\rm out}\).

## Running

A normal run:

```bash
python swssb_n6_continuous_matching_train.py \
    --n-trajectories 5000 \
    --n-times 30 \
    --epochs 150
```

A small smoke test:

```bash
python swssb_n6_continuous_matching_train.py \
    --n-trajectories 100 \
    --n-times 12 \
    --epochs 10
```

The small smoke test may produce too few continuously matched states; this is
expected and does not indicate a physics/code error.

To change the final demo initial state, choose any six-bit state with exactly
three excitations, for example

```bash
--demo-basis 110010
```

## Main outputs

The default output directory is `outputs/`.

Important files are:

- `summary.txt`
- `model_best.pt`
- `dataset_matched.pt`
- `selected_measurements.txt`
- `test_roc.png`
- `test_charge_auc.png`
- `test_probability.png`
- `training_history.png`
- `demo_idealized.png`
- `demo_realistic.png`
- `demo_idealized.csv`
- `demo_realistic.csv`
- `demo_parameters.txt`
