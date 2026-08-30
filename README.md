# ballet-com-cop-analyzer

**A kinematics-only implementation of the COM-to-COP inclination-angle (IA / RCIA) balance
metric, applied to ballet.** Reproduces the method Prof. Tung-Wu Lu's lab uses across its
balance papers, with the only modification being a force-plate-free COP proxy taken from
Lin et al. (2019), so single-leg ballet tasks (releve, passe, pirouette) can be analysed from
ordinary video.

> **What this is.** A pre-enrollment self-study project. It shows I can (1) implement Prof.
> Lu's signature method in code, (2) apply it to ballet, my own domain, and (3) de-risk the
> Plan B research proposal by trying its measurement pipeline ahead of time. Every
> approximation is disclosed and, in notebook 02, quantified. See `docs/LIMITATIONS.md`.
>
> **What this is not.** Not a claim that I can already run Lu-lab research; not muscle-synergy /
> sEMG-NMF (that is separate, Hayashibe-line work); not deep reinforcement learning. Pure
> kinematics.

## What it computes
Video or marker data -> per-frame:
- **COM** via a De Leva (1996) segment-mass model (`balletcomcop/anthropometry.py`),
- **COP proxy** via an ankle-to-toe midpoint during single-leg support, following the approach of
  Lin, Su & Lin 2019 for pirouette (approximated with pose keypoints, not their exact landmarks;
  see `docs/LIMITATIONS.md`) (`balletcomcop/com_cop.py`),
- **IA** = `arcsin(COM-COP vector / |.|)`, faithful to Lu Eq. (1)-(3), split into
  sagittal (anterior-posterior) and frontal (medio-lateral) planes (`balletcomcop/inclination.py`),
- **RCIA** = smoothing-spline time-derivative of IA (open-source GCVSPL analogue),
- AP/ML dominance and ending-phase sway SD (the quantities Plan B / Lin 2019 report).

## Quick start
```bash
pip install -e .                       # installs the balletcomcop package + core deps
# or: pip install -r requirements.txt   # core deps only
pytest                                  # full suite, 100% coverage (no video/mediapipe needed)
python data/make_synthetic.py          # generate the bundled synthetic demo sequence
# then open notebooks/01_video_to_IA_demo.py (jupytext) and run Section A
```
To analyse your own ballet clip: `pip install mediapipe opencv-python`, film one balance task
side-on, set `VIDEO=` in notebook 01 Section B.

## The two-part story (rigor + honesty)
- **L1 (this pipeline, notebook 01):** compute IA/RCIA on my own ballet video, markerless.
- **L2 (done; `scripts/validate_fukuchi.py`, results in `docs/LIMITATIONS.md`):** the COP proxy was
  validated against a force plate on the public Fukuchi 2018 walking dataset (42 adults, subject as
  the unit of analysis). The sagittal IA error is a *deterministic* heel-to-toe stance-phase ramp
  (per-subject mid-stance RMSE 3.9 deg, 95% CI 3.7 to 4.1), and most of the frontal error is an
  artefact of the dataset's lateral-malleolus marker (not the ankle joint centre Lin 2019 used),
  not a property of the proxy itself.

That validation, checked line by line against the raw dataset, is the honest caveat for every
ballet-video result: it shows where the force-plate-free proxy is trustworthy (gross geometry, a
correctable AP structure) and where it is not (the frontal plane on this marker set, fine group
differences, demi-pointe). See `docs/LIMITATIONS.md` for the full interpretation, and note that a
markerless pose estimator would add error on top of these marker-based numbers.

## Repository layout
```
balletcomcop/anthropometry.py   De Leva (1996) segment masses -> whole-body COM        [tested]
balletcomcop/inclination.py     IA = arcsin(COM-COP); AP/ML split; RCIA spline deriv    [tested]
balletcomcop/com_cop.py         COP proxy (Lin 2019 ankle-metatarsal midpoint)          [tested]
balletcomcop/pose_to_keypoints.py  MediaPipe Pose video -> keypoints (lazy import)
balletcomcop/pipeline.py        frames -> COM/COP/IA/RCIA time series
balletcomcop/plotting.py        IA/RCIA and COM-COP-path figures
notebooks/01_video_to_IA_demo.py        L1: synthetic demo + your own video
notebooks/02_forceplate_reproduce.py    L2: exact reproduction + proxy validation
docs/METHOD.md         equations, De Leva, Lin-2019 COP proxy, citations (DOIs)
docs/LIMITATIONS.md    the honest caveat list
tests/                 pytest; runs with the core deps only (no MediaPipe / dataset)
```

## How it maps to Prof. Lu's papers
| Concept | This repo | Lu-lab source |
|---|---|---|
| IA Eq. (1)-(3) | `inclination_cross_product` | Kuo 2022 / Lee 2021 / Yu 2023 |
| COM (segment-weighted) | `anthropometry.compute_com` | 13-segment + Chen 2011 optimization |
| RCIA (smooth + differentiate) | `inclination.rcia` | GCVSPL package |
| single-leg COP without a plate | `com_cop.cop_proxy_single_leg` | Lin, Su & Lin 2019 |
| AP-dominance / ending-phase SD | `inclination.decompose_ap_ml` | Lin 2019; Plan B H1/H2 |

## Citations
- De Leva P. 1996. *J Biomech* 29(9):1223-1230. DOI 10.1016/0021-9290(95)00178-6.
- Kuo C-C et al. 2022. *Sci Rep* 12:2660. DOI 10.1038/s41598-022-06631-8.
- Lee P-A et al. 2021. *Sci Rep* 11:3742. DOI 10.1038/s41598-021-83233-w.
- Yu C-H et al. 2023. *Sensors* 23(22):9040. DOI 10.3390/s23229040.
- Lin C-W, Su F-C, Lin C-F. 2019. *Front Bioeng Biotechnol* 7:290. DOI 10.3389/fbioe.2019.00290.
- Lee H-J, Chou L-S. 2006. *Arch Phys Med Rehabil* 87:569-575 (IA/RCIA origin). DOI 10.1016/j.apmr.2005.11.033.
- Pai Y-C, Patton J. 1997. *J Biomech* 30(4):347-354 (IA-RCIA coupling). DOI 10.1016/s0021-9290(96)00165-0.

## License
MIT (see `LICENSE`).
