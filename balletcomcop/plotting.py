"""Matplotlib views of the IA/RCIA time series (the figures you would show Prof. Lu)."""
from __future__ import annotations

import pandas as pd

# NOTE: matplotlib.pyplot is imported lazily inside each function so importing this package does
# NOT hijack the user's matplotlib backend (the previous module-level matplotlib.use("Agg") did).
# On a headless machine set the env var MPLBACKEND=Agg before running.


def plot_ia_rcia(df: pd.DataFrame, title: str = "Ballet COM-COP balance", savepath: str | None = None):
    """Two-panel figure: IA (sagittal/AP + frontal/ML) and RCIA over time."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(df["time"], df["ia_sagittal"], label="sagittal (AP)")
    ax1.plot(df["time"], df["ia_frontal"], label="frontal (ML)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("Inclination angle (deg)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")

    ax2.plot(df["time"], df["rcia_sagittal"], label="sagittal (AP)")
    ax2.plot(df["time"], df["rcia_frontal"], label="frontal (ML)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("RCIA (deg/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


def plot_com_cop_path(df: pd.DataFrame, savepath: str | None = None):
    """Top-down view of COM and COP-proxy horizontal paths (AP vs ML)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(df["com_ml"], df["com_ap"], label="COM", lw=1.5)
    ax.plot(df["cop_ml"], df["cop_ap"], label="COP proxy", lw=1.0, ls="--")
    ax.set_xlabel("Medio-lateral (m)")
    ax.set_ylabel("Anterior-posterior (m)")
    ax.set_aspect("equal", "box")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
