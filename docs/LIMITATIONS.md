# Limitations (read this honestly, and put it in front of any reviewer)

This repository is a **pre-enrollment self-study tool**, not a validated clinical instrument.
It exists to (a) show I can implement Prof. Lu's COM-COP inclination-angle method in code,
(b) apply it to ballet, my own domain, and (c) de-risk Plan B. Every approximation below is
deliberate and disclosed; quantifying them is the point, not hiding them.

## Measurement limitations
1. **2D / markerless != force-plate 3D.** A single-camera MediaPipe pose gives no true 3D COM
   and no true COP. The depth axis is weakly observed from one view, so medio-lateral IA from a
   single side-view is unreliable. Use front/back view for ML, side view for AP, or 2+ cameras.
2. **COP is a proxy, not a measurement.** The ankle-metatarsal midpoint (Lin et al. 2019) is
   valid only during *single-leg* support and ignores the true pressure distribution. The
   double-leg proxy is cruder still (kinematics cannot recover the load split between feet).
3. **Reduced segment model.** De Leva masses are used, but head/trunk COM are midpoint
   approximations and the hand is folded into the forearm. Lu's lab uses a 13-segment model with
   subject-specific inertial optimization and STA denoising that this does not reproduce.
4. **RCIA smoothing differs from GCVSPL.** A scipy smoothing spline approximates, but is not
   identical to, the GCVSPL package Lu's papers use; the smoothing factor changes RCIA amplitude.

## Additional limitations surfaced by a code and numbers review (2026-06-29)
5. **RCIA on noisy video.** The smoothing spline assumes measurement noise ~10% of the IA SD
   (good for optical mocap, optimistic for markerless single-camera pose). On noisy video the
   derivative can be dominated by pose jitter; report RCIA from video only with the notebook-02
   proxy error attached, and prefer increasing the `smoothing` argument or filtering keypoints
   by MediaPipe `visibility` first (the pipeline does NOT yet visibility-filter).
6. **Foot segment axis.** De Leva's foot CoM ratio is referenced from the heel along the foot
   long axis; this model uses ankle->foot_index, a slightly different axis. Foot is ~1.4% mass,
   so the whole-body COM error is negligible, but it is not the exact De Leva foot placement.
7. **Head CoM is a proxy.** Head CoM is placed at the midpoint of mid_ear->nose (a pose-keypoint
   proxy), NOT on De Leva's vertex-referenced head axis; head is ~7% mass.
8. **Trunk dominates COM error.** The trunk is ~43% of body mass; its midpoint-line CoM
   (De Leva ratio applied to mid_shoulder->mid_hip) is the largest single source of COM
   approximation in this reduced model versus a full 13-segment marker model.

9. **MediaPipe joints are not joint centres.** The De Leva ratios assume segment endpoints at
   anatomical JOINT CENTRES; MediaPipe landmarks are surface/keypoint estimates (a few cm off,
   and the hip/shoulder especially). This adds a systematic, un-quantified COM bias on the video
   path on top of the pose-accuracy noise. Notebook 02's force-plate comparison bounds the
   combined effect.

10. **COP proxy is foot-posture dependent.** Lin et al. 2019 used the ankle-metatarsal midpoint
    for the support foot. That is a fair mid-foot estimate for a FLAT-foot single-leg stance, but
    in a releve / demi-pointe only the ball and toes contact the floor, so the true COP shifts
    anteriorly toward the metatarsal; the midpoint then carries a posterior bias. The proxy is
    therefore most valid for flat-foot single-leg balance and least valid for full demi-pointe.
    Notebook 02's force-plate comparison should be run on a task with a known foot posture.

## Quantified force-plate validation (2026-07-03): what the comparison actually shows

The COP proxy was validated against a force-plate reference on the public Fukuchi et al. (2018)
overground walking dataset (PeerJ 6:e4640; figshare 10.6084/m9.figshare.5722711; CC BY 4.0; 42
adults; markers 150 Hz; five force plates at 300 Hz with COP exported directly). During
single-support the proxy COP was compared with the true COP, and the inclination angle from each
was compared, using the dataset's OWN optical markers. The subject is the unit of analysis
(n = 42); pooled per-frame statistics are descriptive only. An earlier, stronger reading of these
numbers did not survive scrutiny; the findings below are the corrected ones.

1. **The anteroposterior IA error is a deterministic stance-phase ramp, not random noise.** The
   signed sagittal IA error runs from about -1.6 deg at heel strike, through near zero at
   mid-stance, to about +1.2 deg at toe-off, because a fixed mid-foot point cannot follow the true
   COP as it rolls from heel to toe. Per-subject mid-stance sagittal IA RMSE is 3.9 deg (95% CI 3.7
   to 4.1); full-stance is 4.8 deg. Because the error is structured and phase-locked it is
   correctable in principle by a stance-phase or COP-velocity regression; it is NOT an irreducible
   noise floor, and it should not be described as one.

2. **Most of the mediolateral error is a landmark-substitution artefact, not a property of the
   proxy.** This dataset provides only a lateral-malleolus ankle marker, not the ankle joint centre
   that Lin et al. (2019) specified. Using the malleolus injects a systematic mediolateral offset of
   about +35 mm for the right foot and -35 mm for the left (about plus or minus 2.1 deg of frontal
   IA), which cancels to near zero when the two feet are pooled and would otherwise be mistaken for
   near-zero error. The joint centre cannot be reconstructed from a single lateral marker, so this
   dataset cannot faithfully reproduce Lin's proxy in the frontal plane, and the frontal-plane
   number here must not be attributed to Lin's proxy.

3. **This is the COP proxy error alone, computed from optical markers.** A markerless single-camera
   pose estimator, the intended input for the ballet application, would add its own error on top and
   cannot reduce these values.

4. **This is walking, not balance, and not demi-pointe.** Quasi-static single-leg balance, where the
   COP is nearly stationary, should give a smaller anteroposterior error than walking. But the
   pirouette posture is on releve or demi-pointe, where only the forefoot contacts the floor and the
   true COP shifts anteriorly of the ankle-metatarsal midpoint: a systematic bias in the opposite
   sense to the flat-foot case, and untested here. The balance-relevant and demi-pointe accuracy
   cannot be established without synchronised force-plate and foot-marker data during single-leg
   stance, which no public dataset provides (the one public single-leg-stance dataset found, OLST on
   PhysioNet, marks no point on the foot) and which is the necessary next measurement.

5. **The validated construction is not identical to the shipped one.** This validation used the
   dataset's lateral-malleolus marker and the midpoint of its MT1 and MT5 markers, with a
   pelvis-marker-centroid COM. The package's `cop_proxy_single_leg` instead uses the ankle and the
   foot-index (toe) pose keypoints, and `compute_com` uses a De Leva whole-body model. The
   qualitative finding (a deterministic AP stance-phase ramp) transfers, but the exact degree values
   above characterise the validated construction, not the precise foot-index and De-Leva-COM pipeline
   the code runs. Lin et al. (2019) specified a single metatarsal marker and the ankle joint centre;
   both the validation and the shipped code only approximate that with the landmarks available, so
   neither reproduces Lin's exact proxy, and the word "exactly" should not be used for either.

In short: the tool reproduces the published method faithfully in code, and the force-plate
comparison shows where the force-plate-free proxy is trustworthy (gross geometry; a correctable AP
structure) and where it is not (the frontal plane on this marker set; fine group differences;
demi-pointe). Do not report a ballet IA value as accurate to better than these bounds, and do not
present the walking numbers as if they were the balance-task accuracy.

## What this is NOT
- Not a claim that I can already do Lu-lab research; it is self-study toward Plan B.
- Not muscle-synergy / sEMG-NMF (that is the separate Hayashibe-line work and is Tier-2 for Lu).
- Not deep reinforcement learning. Pure kinematics only.
