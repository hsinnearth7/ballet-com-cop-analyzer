# data/

- `make_synthetic.py` generates `sample_synthetic.csv` (a fake single-leg-balance sequence)
  so the pipeline, tests, and notebook 01 run with no downloads. **It is not real data.**

## Open datasets for notebook 02 (download yourself; do NOT commit large files)
| Dataset | Contents | Access |
|---|---|---|
| Fukuchi et al. 2018 | walking, mocap + force plates + kinetics | PeerJ 6:e4640; figshare DOI 10.6084/m9.figshare.5722711 |
| Camargo et al. 2021 ("Gait120") | level/stairs/ramp/STS, mocap + force plates + EMG | Nature Sci Data |
| OLST one-legged-stand | single-leg stance, mocap + force plate | Nature Sci Data |
| Manfrim et al. 2025 (ballet turnout) | lower-limb turnout kinematics | PeerJ + figshare |

`.gitignore` excludes `*.c3d *.mat *.mp4 *.mov` and large CSVs so the repo stays small.
