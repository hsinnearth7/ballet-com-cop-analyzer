import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe for CI

import numpy as np
import pandas as pd

from balletcomcop.plotting import plot_com_cop_path, plot_ia_rcia


def _demo_df(n=20):
    t = np.linspace(0, 1, n)
    return pd.DataFrame({
        "time": t,
        "ia_sagittal": np.sin(t), "ia_frontal": np.cos(t),
        "rcia_sagittal": np.cos(t), "rcia_frontal": -np.sin(t),
        "com_ap": t, "com_ml": 0.1 * t, "cop_ap": 0.0 * t, "cop_ml": 0.0 * t,
    })


def test_plot_ia_rcia_returns_figure():
    from matplotlib.figure import Figure
    fig = plot_ia_rcia(_demo_df(), savepath=None)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_com_cop_path_returns_figure():
    from matplotlib.figure import Figure
    fig = plot_com_cop_path(_demo_df(), savepath=None)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_saves_file():
    # Write into the repo dir (this machine's %TEMP%/pytest-of-USER is access-denied), then clean up.
    out = "_test_ia_output.png"
    try:
        import matplotlib.pyplot as plt
        fig = plot_ia_rcia(_demo_df(), savepath=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
        plt.close(fig)
        os.remove(out)
        fig2 = plot_com_cop_path(_demo_df(), savepath=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
        plt.close(fig2)
    finally:
        if os.path.exists(out):
            os.remove(out)
