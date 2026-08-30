# Method: COM-COP inclination angle (IA) and its rate of change (RCIA)

## 1. The Lu-lab definition (what we reproduce)
The inclination angle is the angle of the body centre-of-mass (COM) relative to the centre
of pressure (COP), measured from vertical. Across the Lu-lab papers the equations are:

```
P = COM - COP                       (COM-to-COP vector)
t = Z x (P / |P|)                   Eq. 1   (Z = vertical unit vector)
sagittal IA = arcsin(t_Y)           Eq. 2   (anterior-posterior plane)
frontal  IA = +/- arcsin(t_X)       Eq. 3   (medio-lateral plane)
RCIA = d(IA)/dt                             (rate of change, deg/s)
```

With `Z = (0,0,1)` the cross product gives `t = (-P_y, P_x, 0)`, so
`sagittal IA = arcsin(P_x / |P|)` and `frontal IA = arcsin(-P_y / |P|)`. We implement this
exactly in `balletcomcop/inclination.py::inclination_cross_product`. The axis-labelled form
`inclination_angles(com, cop, ap_axis, ml_axis, vert_axis)` is **sign-identical** (frontal =
`arcsin(-P_ml/|P|)`, matching Eq. 3), verified by `test_cross_product_and_axis_form_agree`, and
is what the pipeline uses. The De Leva trunk (0.5138 M / 0.4964 F) and shank (0.4395 / 0.4352)
CoM ratios were verified against the Visual3D / HAS-Motion tabulation of De Leva (1996).

Sources (deep-noted full text in the lab folder `Papers/`):
- Kuo C-C et al. 2022, *Sci Rep* 12:2660. DOI 10.1038/s41598-022-06631-8.
- Lee P-A et al. 2021, *Sci Rep* 11:3742. DOI 10.1038/s41598-021-83233-w.
- Yu C-H et al. 2023, *Sensors* 23(22):9040. DOI 10.3390/s23229040.
- Origin of the IA/RCIA metric: Lee H-J & Chou L-S 2006, *Arch Phys Med Rehabil* 87:569-575. DOI 10.1016/j.apmr.2005.11.033.
- Position-velocity (IA-RCIA coupling) framework: Pai Y-C & Patton J 1997, *J Biomech* 30:347-354. DOI 10.1016/s0021-9290(96)00165-0.

### 1a. What `arcsin(component/|P|)` actually measures (verified 2026-06-29)
Lu's IA is `arcsin(P_component / |P|)`, i.e. the angle of P above the plane orthogonal to that
component axis, NOT the planar projection angle `arctan2(P_ap, P_vert)`. The two are identical at
the small inclinations of balance (diff < 0.03 deg at 5 deg, 0.67 deg at 16 deg) but diverge at
large angles. A consequence: the sagittal and frontal IA are **coupled through |P|** (adding ML
lean shrinks the reported sagittal IA), so they are not independent planar projections. This is
Lu's published definition and is what we reproduce; both `inclination_cross_product` and the
independent `arctan2(P_i, hypot(other two))` reference agree exactly (test
`test_ia_matches_independent_atan2_reference`).

## 2. COM from kinematics (no force plate)
COM = mass-weighted sum of segmental COMs (`balletcomcop/anthropometry.py`). Segment **masses** are
the De Leva (1996) values (*J Biomech* 29:1223-1230, DOI 10.1016/0021-9290(95)00178-6); they sum
to 1.0 per sex (enforced by a test).
Limb COM longitudinal ratios are De Leva; head and trunk COM are placed at the segment midpoint
as a documented simplification (no vertex / suprasternale landmark from a pose estimator). The
hand mass is folded into the forearm because pose estimators have no knuckle landmark.

Lu's lab uses a 13-segment model with a subject-specific inertial-parameter optimization
(Chen S-C et al. 2011, *Gait Posture* 33:695-700, DOI 10.1016/j.gaitpost.2011.03.004) and
skin-marker (STA) global-optimization denoising. Our reduced pose-keypoint model is a portfolio-grade approximation of that pipeline.

## 3. COP proxy (the key trick that makes single-leg ballet kinematics-friendly)
A force plate is normally required for COP. But Lin, Su & Lin (2019, *Front Bioeng Biotechnol*
7:290), who applied the **same** COM-COP method to pirouette, approximate the COP during
single-leg phases as the **midpoint of the ankle joint centre and the metatarsal marker**.
We reuse exactly that (`balletcomcop/com_cop.py::cop_proxy_single_leg`). This is why single-leg ballet
tasks (releve, passe, pirouette) can be analysed from kinematics alone.

## 4. RCIA via smoothing spline (open-source GCVSPL analogue)
Lu's papers smooth and differentiate IA with the GCVSPL package. We use scipy
`UnivariateSpline` (cubic) with a generalized-cross-validation-like smoothing factor
(`balletcomcop/inclination.py::rcia`), defaulting to `s = n*(0.1*SD(IA))^2` (assumes ~10% measurement
noise) and returning zeros for a constant signal. Substitute GCVSPL when reproducing exact
published curves in notebook 02.

## 5. Coordinate frames (read before using your own video)
The IA functions take explicit `ap_axis`, `ml_axis`, `vert_axis` so you can map whatever frame
your data uses. MediaPipe image landmarks have y growing downward (we flip to vertical-up);
world landmarks are metric. For a side-on single camera, the camera depth axis (toward/away)
is poorly observed, so the medio-lateral IA from a single side-view is unreliable - prefer a
front-or-back view for ML, a side view for AP, or two cameras (Pose2Sim) for full 3D.
