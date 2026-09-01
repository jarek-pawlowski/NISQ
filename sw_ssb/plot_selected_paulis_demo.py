#!/usr/bin/env python3
"""
Plot the K Pauli observables selected by the trained SW-like witness
for the same idealized / realistic demo trajectories.

Place this file next to utils.py, or make sure utils.py is importable.

Example:
    python plot_selected_paulis_demo.py \
        --model-dir outputs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils as u


def load_model(model_dir):
    ckpt = torch.load(
        Path(model_dir) / "model_best.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = u.SWWitness(
        n_features=len(u.PAULI_LABELS),
        K=int(ckpt["K"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mean = np.asarray(ckpt["mean"]).reshape(-1)
    std = np.asarray(ckpt["std"]).reshape(-1)
    selected_idx = np.asarray(ckpt["selected_indices"], dtype=int)
    selected_labels = list(ckpt["selected_labels"])

    return model, mean, std, selected_idx, selected_labels


def evolve_case(pars, rho0, times):
    H = u.build_hamiltonian(
        pars["Jxy"],
        pars["Jz"],
        pars["h"],
    )
    L = u.liouvillian(
        H,
        pars["gamma_phi"],
        pars["gamma_hop"],
        pars["gamma_loss"],
        pars["gamma_gain"],
    )
    return u.evolve_at_times(L, rho0, times)


@torch.no_grad()
def analyze_case(
    name,
    pars,
    model,
    mean,
    std,
    selected_idx,
    selected_labels,
    rho0,
    times,
    output_dir,
):
    states = evolve_case(
        pars,
        rho0,
        times,
    )

    features = np.asarray([
        u.local_pauli_features(rho)
        for rho in states
    ])

    selected = features[:, selected_idx]

    x = (
        features - mean[None, :]
    ) / std[None, :]

    logits, _ = model(
        torch.tensor(
            x,
            dtype=torch.float32,
        )
    )
    p_sw = torch.sigmoid(logits).cpu().numpy()

    # Selected Pauli expectation values.
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for j, label in enumerate(
        selected_labels
    ):
        ax.plot(
            times,
            selected[:, j],
            label=label,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\langle P_i\rangle$")
    ax.set_title(
        f"{name}: selected Pauli observables"
    )
    ax.legend(
        ncol=2,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"demo_selected_paulis_{name}.png",
        dpi=180,
    )
    plt.close(fig)

    # Witness alone, for direct comparison.
    fig, ax = plt.subplots(
        figsize=(7.2, 3.8)
    )
    ax.plot(
        times,
        p_sw,
        label=r"$P_{\rm SW}$",
    )
    ax.axhline(
        0.5,
        linestyle="--",
        label="decision threshold",
    )
    ax.set_xlabel("time")
    ax.set_ylabel(r"$P_{\rm SW}$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(name)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"demo_witness_only_{name}.png",
        dpi=180,
    )
    plt.close(fig)

    arr = np.column_stack(
        [times, selected, p_sw]
    )
    header = (
        "time,"
        + ",".join(selected_labels)
        + ",P_SW"
    )
    np.savetxt(
        output_dir
        / f"demo_selected_paulis_{name}.csv",
        arr,
        delimiter=",",
        header=header,
        comments="",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "outputs"
        ),
    )
    parser.add_argument(
        "--bitstring",
        type=str,
        default="101010",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--n-times",
        type=int,
        default=201,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=u.DEFAULT_SEED + 100,
    )

    args = parser.parse_args()
    args.model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        model,
        mean,
        std,
        selected_idx,
        selected_labels,
    ) = load_model(
        args.model_dir
    )

    print(
        "selected observables:",
        ", ".join(selected_labels),
    )

    # rho0 = u.basis_state_rho(
    #     args.bitstring
    # )
    rng = np.random.default_rng(args.seed)
    rho0 = u.random_fixed_sector_rho(
        rng
    )
    
    times = np.linspace(
        0.0,
        args.t_max,
        args.n_times,
    )

    cases = {
        "idealized":
            u.ideal_demo_parameters(),
        "realistic":
            u.realistic_demo_parameters(
                args.seed
            ),
    }

    for name, pars in cases.items():
        analyze_case(
            name=name,
            pars=pars,
            model=model,
            mean=mean,
            std=std,
            selected_idx=selected_idx,
            selected_labels=selected_labels,
            rho0=rho0,
            times=times,
            output_dir=args.model_dir,
        )

    print(
        "saved to:",
        args.model_dir.resolve(),
    )


if __name__ == "__main__":
    main()
