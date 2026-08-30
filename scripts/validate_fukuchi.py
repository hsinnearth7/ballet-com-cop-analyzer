"""Reproduce the force-plate validation of the COP proxy on the Fukuchi et al. (2018) dataset.

This regenerates the numbers reported in docs/LIMITATIONS.md ("Quantified force-plate
validation") from the raw public data.

Data (about 586 MB, Creative Commons Attribution 4.0):
    Fukuchi CA, Fukuchi RK, Duarte M. A public dataset of overground and treadmill walking
    kinematics and kinetics in healthy individuals. PeerJ 6:e4640 (2018).
    figshare 10.6084/m9.figshare.5722711 ; ASCII archive:
    https://ndownloader.figshare.com/files/10058986  (WBDSascii.zip, md5 ad9e6311d9b84b53acaf58d114d51c6d)

Usage:
    python scripts/validate_fukuchi.py path/to/WBDSascii.zip [max_subject]

Method (see docs/LIMITATIONS.md for the honest interpretation):
    Lab frame X = anterior-posterior (progression), Y = vertical, Z = medio-lateral.
    Vertical GRF is Fy. A single-support instant is a frame with exactly one force plate
    carrying Fy above 50 N. The support foot is chosen by heel-marker proximity to the true COP
    (independent of the ankle and metatarsal markers that form the proxy). The COP proxy is the
    midpoint of the ankle marker and the midpoint of MT1 and MT5, projected to the floor. IA is
    computed with the package's own inclination_angles, using a pelvis-marker-centroid COM.
    The subject is the unit of analysis (n = 42); pooled per-frame values are descriptive.

    IMPORTANT: the ankle marker here is the lateral malleolus, not the ankle joint centre that
    Lin et al. (2019) specified; the dataset has no medial marker to reconstruct the joint centre,
    so the frontal-plane result carries a per-foot lateral-malleolus offset (see LIMITATIONS.md).
"""
from __future__ import annotations

import re
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from balletcomcop.inclination import inclination_angles

FV_THRESH = 50.0
MAX_FOOT_COP_DIST = 300.0
MID_LO, MID_HI = 0.3, 0.7
MIN_CONTACT = 10
AP, VT, ML = 0, 1, 2

MKR_RE = re.compile(r"^WBDS(\d+)walkO(\d+)([CFS])mkr\.txt$")
FUNNEL = {"frames": 0, "one_plate": 0, "on_foot": 0}


def _mxyz(df: pd.DataFrame, m: str) -> np.ndarray:
    return df[[m + "X", m + "Y", m + "Z"]].to_numpy(dtype=float)


def analyze_trial(mkr: pd.DataFrame, grf: pd.DataFrame) -> list[tuple]:
    nm, ng = len(mkr), len(grf)
    if nm < 2 or ng < 2:
        return []
    fv = [grf[f"Fy{p}"].to_numpy(float) for p in range(1, 6)]
    cx = [grf[f"COPx{p}"].to_numpy(float) for p in range(1, 6)]
    cz = [grf[f"COPz{p}"].to_numpy(float) for p in range(1, 6)]
    Ra, R1, R5, Rh = (_mxyz(mkr, "R.Ankle"), _mxyz(mkr, "R.MT1"), _mxyz(mkr, "R.MT5"), _mxyz(mkr, "R.Heel"))
    La, L1, L5, Lh = (_mxyz(mkr, "L.Ankle"), _mxyz(mkr, "L.MT1"), _mxyz(mkr, "L.MT5"), _mxyz(mkr, "L.Heel"))
    pelvis = 0.25 * (_mxyz(mkr, "R.ASIS") + _mxyz(mkr, "L.ASIS") + _mxyz(mkr, "R.PSIS") + _mxyz(mkr, "L.PSIS"))

    keys, com, ct, cp, mm = [], [], [], [], []
    for i in range(nm):
        FUNNEL["frames"] += 1
        j = round(i * (ng - 1) / (nm - 1))
        loaded = [p for p in range(5) if fv[p][j] > FV_THRESH]
        if len(loaded) != 1:
            continue
        FUNNEL["one_plate"] += 1
        p = loaded[0]
        cop = np.array([cx[p][j], cz[p][j]])
        rh = np.array([Rh[i][AP], Rh[i][ML]])
        lh = np.array([Lh[i][AP], Lh[i][ML]])
        dR, dL = np.linalg.norm(rh - cop), np.linalg.norm(lh - cop)
        if min(dR, dL) > MAX_FOOT_COP_DIST:
            continue
        FUNNEL["on_foot"] += 1
        if dR <= dL:
            foot, ank, mt = "R", Ra[i], 0.5 * (R1[i] + R5[i])
        else:
            foot, ank, mt = "L", La[i], 0.5 * (L1[i] + L5[i])
        proxy = 0.5 * (ank + mt)
        e_ap, e_ml = proxy[AP] - cop[0], proxy[ML] - cop[1]
        com_i = pelvis[i]
        cop_true = np.array([cop[0], 0.0, cop[1]])
        cop_prox = np.array([proxy[AP], 0.0, proxy[ML]])
        if not np.all(np.isfinite(np.concatenate([com_i, cop_true, cop_prox, [e_ap, e_ml]]))) or com_i[VT] <= 0:
            continue
        keys.append((i, p, foot))
        com.append(com_i)
        ct.append(cop_true)
        cp.append(cop_prox)
        mm.append((e_ap, e_ml))
    if not keys:
        return []

    com_a, ct_a, cp_a, mm_a = np.array(com), np.array(ct), np.array(cp), np.array(mm)
    sag_t, fro_t = inclination_angles(com_a, ct_a, ap_axis=AP, ml_axis=ML, vert_axis=VT)
    sag_p, fro_p = inclination_angles(com_a, cp_a, ap_axis=AP, ml_axis=ML, vert_axis=VT)
    ia_sag_err, ia_fro_err = sag_p - sag_t, fro_p - fro_t

    rows: list[tuple] = []
    contact: list[int] = []

    def flush(c: list[int]) -> None:
        L = len(c)
        for idx, k in enumerate(c):
            mid = L >= MIN_CONTACT and (MID_LO * L <= idx <= MID_HI * L)
            phase = idx / (L - 1) if L > 1 else 0.0
            foot = 1.0 if keys[k][2] == "R" else -1.0
            rows.append((mm_a[k, 0], mm_a[k, 1], ia_sag_err[k], ia_fro_err[k], phase, mid, foot))

    for k in range(len(keys)):
        if contact and keys[k][0] == keys[contact[-1]][0] + 1 and keys[k][1] == keys[contact[-1]][1] and keys[k][2] == keys[contact[-1]][2]:
            contact.append(k)
        else:
            if contact:
                flush(contact)
            contact = [k]
    if contact:
        flush(contact)
    return rows


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def main(zip_path: str, max_subj: int) -> None:
    zf = zipfile.ZipFile(zip_path)
    names = set(zf.namelist())
    trials = sorted((int(m.group(1)), n, n.replace("mkr.txt", "grf.txt"))
                    for n in names for m in [MKR_RE.match(n)] if m and int(m.group(1)) <= max_subj)
    per_subject: dict[int, list] = {}
    for subj, mkr_name, grf_name in trials:
        if grf_name not in names:
            continue
        with zf.open(mkr_name) as fh:
            mkr = pd.read_csv(fh, sep="\t")
        with zf.open(grf_name) as fh:
            grf = pd.read_csv(fh, sep="\t")
        rows = analyze_trial(mkr, grf)
        if rows:
            per_subject.setdefault(subj, []).extend(rows)

    def subj_ci(col: int, mid_only: bool) -> str:
        vals = [_rmse(np.array([x for x in rows if (x[5] if mid_only else True)], dtype=float)[:, col])
                for rows in per_subject.values()
                if any((x[5] if mid_only else True) for x in rows)]
        v = np.array(vals)
        n = len(v)
        m, sd = float(v.mean()), float(v.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
        return f"{m:6.2f} (95% CI {m - half:.2f} to {m + half:.2f}, n={n})"

    allr = np.array([r for rows in per_subject.values() for r in rows], dtype=float)
    print(f"subjects={len(per_subject)}  single_support_frames={len(allr)}")
    print("PER-SUBJECT RMSE (subject is the unit of analysis):")
    for mid_only, wtag in [(False, "FULL"), (True, "MID ")]:
        print(f"  [{wtag}] COP AP mm : {subj_ci(0, mid_only)}")
        print(f"  [{wtag}] COP ML mm : {subj_ci(1, mid_only)}")
        print(f"  [{wtag}] IA  AP deg: {subj_ci(2, mid_only)}")
        print(f"  [{wtag}] IA  ML deg: {subj_ci(3, mid_only)}")
    print("STANCE-PHASE PROFILE (signed means; a deterministic ramp, not random noise):")
    ph = allr[:, 4]
    for b in range(10):
        m = (ph >= b / 10) & (ph < (b + 1) / 10)
        if m.sum():
            print(f"  phase {b*10:3d}-{b*10+10:3d}%: IA_AP_err={allr[m,2].mean():+5.2f} deg")
    print("SIGNED BIAS BY FOOT (near-zero pooled ML bias is a cancellation artefact):")
    for fsign, flab in [(1.0, "R"), (-1.0, "L")]:
        m = allr[:, 6] == fsign
        if m.sum():
            print(f"  foot {flab}: COP_ML_bias={allr[m,1].mean():+6.1f} mm  IA_ML_bias={allr[m,3].mean():+5.2f} deg")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 42)
