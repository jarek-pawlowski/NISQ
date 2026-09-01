#!/usr/bin/env python3
"""
Train a sparse six-qubit SW-like witness.

The physics, matching, neural network and diagnostics live in utils.py.
This file is deliberately kept as the readable experiment pipeline.
"""

import argparse
from pathlib import Path

import numpy as np

import utils as u


def parse_args():
    p = argparse.ArgumentParser()

    # Dataset / physical family
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--n-trajectories", type=int, default=1500)
    p.add_argument("--n-times", type=int, default=20)
    p.add_argument("--t-max", type=float, default=10.0)
    p.add_argument("--regime-a-fraction", type=float, default=0.50)
    p.add_argument("--late-time-fraction", type=float, default=0.70)
    p.add_argument("--late-time-start", type=float, default=5.0)

    # Continuous charge matching
    p.add_argument("--match-eps-pout", type=float, default=0.01)
    p.add_argument("--match-eps-varq", type=float, default=0.05)
    p.add_argument("--match-eps-qmean", type=float, default=0.03)
    p.add_argument("--match-k-neighbors", type=int, default=64)

    # Learned witness
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=u.DEFAULT_SEED)

    # Final fixed-state sanity check
    p.add_argument("--demo-basis", type=str, default="101010")
    p.add_argument("--demo-times", type=int, default=101)

    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("N=6 CONTINUOUSLY MATCHED LEARNED SW-LIKE WITNESS")
    print("=" * 72)
    print(f"candidate observables: {len(u.PAULI_LABELS)} one-/two-body Paulis")

    # 1. Generate the restricted Lindbladian family.
    raw = u.generate_raw_dataset(
        n_trajectories=args.n_trajectories,
        n_times=args.n_times,
        t_max=args.t_max,
        seed=args.seed,
        regime_a_fraction=args.regime_a_fraction,
        late_time_fraction=args.late_time_fraction,
        late_time_start=args.late_time_start,
    )

    # 2. Split by trajectories, impose C_lin < eps_C, and match opposite
    #    R2 classes at nearly identical charge-sector mixing.
    matched, train_idx, val_idx, test_idx, match_info = u.prepare_matched_dataset(
        raw,
        eps_pout=args.match_eps_pout,
        eps_varq=args.match_eps_varq,
        eps_qmean=args.match_eps_qmean,
        k_neighbors=args.match_k_neighbors,
        seed=args.seed,
    )

    print()
    print(
        f"conditioned: {match_info['n_conditioned']} states | "
        f"R2-={match_info['conditioned_class0']} | "
        f"R2+={match_info['conditioned_class1']}"
    )
    for split in ("train", "val", "test"):
        info = match_info[split]
        print(
            f"{split:5s}: {info['pairs']:4d} matched pairs | "
            f"<|d p_out|>={info['mean_abs_dpout']:.4g} | "
            f"<|d VarQ|>={info['mean_abs_dvarq']:.4g} | "
            f"<|d<Q>|>={info['mean_abs_dqmean']:.4g}"
        )

    if len(matched["y"]) < 100:
        raise RuntimeError(
            "Too few matched states. Increase --n-trajectories or relax matching."
        )

    # 3. Normalize using the training split only.
    X, y, mean, std = u.normalize_matched_features(matched, train_idx)

    # 4. Jointly learn the hard Top-K selector and MLP reader.
    model, history, best_val_auc = u.train_model(
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        K=args.K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 3,
    )

    # 5. Frozen test set and charge-only baselines.
    test = u.evaluate_model(model, X[test_idx], y[test_idx])
    charge_aucs = u.charge_baselines(matched, test_idx)
    selected_idx, selected_labels = u.selected_observables(model)

    # 6. Save model, dataset, and all routine diagnostics.
    u.save_checkpoint(
        model, mean, std, selected_idx, selected_labels,
        args.K, args, args.output
    )
    u.save_matched_dataset(matched, train_idx, val_idx, test_idx, args.output)
    u.save_standard_diagnostics(
        history, test, *charge_aucs, output_dir=args.output
    )
    u.save_selected_measurements(selected_idx, selected_labels, args.output)
    u.write_summary(
        args, raw, matched, match_info, best_val_auc, test,
        charge_aucs, selected_labels, args.output
    )

    # 7. Final selected-state demonstration.
    #    Same |101010> initial state, two evolutions:
    #      (i) idealized homogeneous model,
    #      (ii) one realistic/disordered member of the learned family.
    u.run_final_demo(
        model=model,
        mean=mean,
        std=std,
        output_dir=args.output,
        t_max=args.t_max,
        n_times=args.demo_times,
        bitstring=args.demo_basis,
        seed=args.seed + 100,
    )

    q_auc, var_auc, pout_auc = charge_aucs
    print()
    print("=" * 72)
    print("FINAL")
    print("=" * 72)
    print(f"test AUC = {test['auc']:.4f}")
    print(f"test ACC = {test['acc']:.4f}")
    print(
        f"charge-only AUCs: <Q>={q_auc:.4f}, "
        f"VarQ={var_auc:.4f}, p_out={pout_auc:.4f}"
    )
    print("selected:", ", ".join(selected_labels))
    print("demo plots: demo_idealized.png, demo_realistic.png")
    print("saved to:", args.output.resolve())


if __name__ == "__main__":
    main()
