# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # L2: force-plate validation of the COP proxy (completed)
#
# This validation has been run. The kinematics-only COP proxy was compared against a true
# force-plate COP on the public Fukuchi et al. (2018) overground walking dataset (42 adults,
# markers plus directly exported COP, CC BY 4.0), and the resulting inclination angle (IA) was
# compared with the force-plate IA, using the dataset's own optical markers. The full analysis
# lives in `scripts/validate_fukuchi.py`, and the honest interpretation is in
# `docs/LIMITATIONS.md` (section "Quantified force-plate validation").
#
# ## What it found (subject is the unit of analysis, n = 42)
# - The sagittal (anterior-posterior) IA error is a **deterministic heel-to-toe stance-phase ramp**
#   (about -1.6 deg at heel strike to +1.2 deg at toe-off), per-subject mid-stance RMSE 3.9 deg
#   (95% CI 3.7 to 4.1). It is structured and correctable in principle, not random noise.
# - Most of the frontal (medio-lateral) error is a **landmark artefact**: this dataset carries only a
#   lateral-malleolus marker, not the ankle joint centre Lin et al. (2019) specified, which injects a
#   systematic plus or minus 35 mm per-foot offset that cancels when the feet are pooled. This is not
#   a property of Lin's proxy and cannot be reproduced faithfully on this marker set.
# - These numbers isolate the COP proxy (computed from markers); a markerless pose estimator would add
#   its own error. They are from walking, not from quasi-static single-leg balance or demi-pointe,
#   whose accuracy needs foot-marked single-leg force-plate data that no public dataset provides.
#
# ## Reproduce it
# 1. Download the Fukuchi et al. (2018) ASCII archive (about 586 MB, CC BY 4.0):
#    `https://ndownloader.figshare.com/files/10058986` (WBDSascii.zip,
#    md5 `ad9e6311d9b84b53acaf58d114d51c6d`).
# 2. Run the reproduction script from the repository root:

# %%
REPRODUCE = "python scripts/validate_fukuchi.py path/to/WBDSascii.zip"
print("Reproduce the force-plate validation with:\n  " + REPRODUCE)
print("Then read docs/LIMITATIONS.md for the honest interpretation of the numbers it prints.")
