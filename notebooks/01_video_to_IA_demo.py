# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # L1 demo: Ballet COM-COP inclination angle (IA / RCIA)
#
# This notebook computes Prof. Lu's COM-to-COP inclination angle (IA) and its rate of change
# (RCIA) for a single-leg balance task, **from kinematics only** (no force plate).
#
# - **Section A** runs on bundled *synthetic* data so the notebook works out of the box.
# - **Section B** runs on **your own ballet video** via MediaPipe Pose (install + film required).
#
# Method recap (see `docs/METHOD.md`): COM = De Leva (1996) segment-weighted sum of joint
# positions; COP proxy = midpoint of ankle and metatarsal of the support foot (Lin et al. 2019);
# IA = arcsin of the normalized COM-COP vector; RCIA = smoothing-spline derivative of IA.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import pandas as pd
from balletcomcop.pose_to_keypoints import frame_to_kp_dict, keypoints_from_video
from balletcomcop.pipeline import analyze_frames
from balletcomcop.plotting import plot_ia_rcia, plot_com_cop_path
from balletcomcop.inclination import decompose_ap_ml

# %% [markdown]
# ## Section A: synthetic single-leg-balance demo (no video needed)

# %%
df = pd.read_csv("../data/sample_synthetic.csv")
frames = sorted(df["frame"].unique())
# Synthetic frame is x=anterior-posterior, y=medio-lateral, z=vertical(up).
kp_by_frame = {f: frame_to_kp_dict(df, f, axes=("x", "y", "z")) for f in frames}
res = analyze_frames(kp_by_frame, fps=60.0, sex="female", support="right",
                     ap_axis=0, ml_axis=1, vert_axis=2)
res[["time", "ia_sagittal", "ia_frontal", "rcia_sagittal", "rcia_frontal"]].head()

# %%
fig = plot_ia_rcia(res, title="Synthetic releve: COM-COP IA / RCIA",
                   savepath="../docs/example_output.png")
summary = decompose_ap_ml(res["ia_sagittal"].to_numpy(), res["ia_frontal"].to_numpy())
print("AP/ML summary:", {k: round(v, 3) for k, v in summary.items()})
plot_com_cop_path(res)

# %% [markdown]
# Read the AP-dominance ratio the way Plan B / Lin et al. 2019 do: a higher `ap_dominance`
# and a smaller `ap_sd` (ending-phase sway SD) is the experienced-dancer signature.

# %% [markdown]
# ## Section B: your own ballet video (MediaPipe)
#
# 1. `pip install mediapipe opencv-python`
# 2. Film one balance task (releve / passe / single turn) from a fixed side-on camera.
# 3. Set `VIDEO` below and run. Then choose which DataFrame axis is anterior-posterior,
#    medio-lateral, vertical for *your* camera angle (this matters; see docs/METHOD.md).
#
# Honest limitation: a single 2D camera cannot recover true 3D COM or true COP. Treat the
# output as a **proxy / feasibility demo**, validated against ground truth in notebook 02.

# %%
VIDEO = None  # e.g. "../data/my_releve.mp4"
if VIDEO:
    kdf = keypoints_from_video(VIDEO, use_world=True, model_complexity=2)
    frames = sorted(kdf["frame"].unique())
    # keypoints_from_video stores x=image-horizontal, y=depth, z=vertical(up-positive).
    # SIDE-on camera -> ("x","y","z"); FRONT/BACK camera -> ("y","x","z"). support=None now
    # auto-detects the lower (support) foot correctly because z is up-positive.
    kp_by_frame = {f: frame_to_kp_dict(kdf, f, axes=("x", "y", "z")) for f in frames}
    res = analyze_frames(kp_by_frame, fps=30.0, sex="female", support=None)
    plot_ia_rcia(res, title="My releve: COM-COP IA / RCIA")
else:
    print("Set VIDEO to your own clip to run Section B.")
