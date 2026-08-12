"""Test keypoints_from_video by mocking cv2 + mediapipe (no real video / install needed).

This finally exercises the R3 fixes: cap.isOpened() bad-path guard, the world->image fallback,
and the dropped-frame warning -- none of which were covered before.
"""
import sys
from types import SimpleNamespace

import pytest

from balletcomcop.pose_to_keypoints import MP_INDEX_TO_NAME, keypoints_from_video


def _landmarks(yval=0.5):
    # 33 MediaPipe landmarks; give each a known y so we can check the -lm.y vertical flip.
    lm = [SimpleNamespace(x=0.1 * i, y=yval, z=0.2, visibility=0.9) for i in range(33)]
    return SimpleNamespace(landmark=lm)


def _fake_cv2(n_frames, opened=True):
    class Cap:
        def __init__(self, path):
            self._i = 0
        def isOpened(self):
            return opened
        def read(self):
            if self._i < n_frames:
                self._i += 1
                return True, object()
            return False, None
        def release(self):
            pass
    return SimpleNamespace(VideoCapture=Cap, cvtColor=lambda frame, code: frame, COLOR_BGR2RGB=0)


def _fake_mp(per_frame):
    class Pose:
        def __init__(self, **kw):
            self._i = 0
        def process(self, img):
            w, im = per_frame[self._i]
            self._i += 1
            return SimpleNamespace(pose_world_landmarks=w, pose_landmarks=im)
        def close(self):
            pass
    return SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(Pose=Pose)))


def _install(monkeypatch, n_frames, per_frame, opened=True):
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(n_frames, opened))
    monkeypatch.setitem(sys.modules, "mediapipe", _fake_mp(per_frame))


def test_bad_path_raises(monkeypatch):
    _install(monkeypatch, n_frames=0, per_frame=[], opened=False)
    with pytest.raises(FileNotFoundError, match="could not open"):
        keypoints_from_video("nonexistent.mp4")


def test_happy_path_vertical_flip(monkeypatch):
    lms = _landmarks(yval=0.5)
    _install(monkeypatch, n_frames=2, per_frame=[(lms, lms), (lms, lms)])
    df = keypoints_from_video("clip.mp4")
    assert df["frame"].nunique() == 2
    assert set(df["keypoint"]) == set(MP_INDEX_TO_NAME.values())
    assert (df["z"] == -0.5).all()  # z = -lm.y (vertical up-positive)


def test_world_none_falls_back_to_image(monkeypatch):
    lms = _landmarks()
    _install(monkeypatch, n_frames=2, per_frame=[(None, lms), (lms, lms)])
    df = keypoints_from_video("clip.mp4", use_world=True, fallback_to_image=True)
    assert df["frame"].nunique() == 2  # frame 0 recovered via image fallback


def test_no_pose_frame_is_dropped_with_warning(monkeypatch):
    lms = _landmarks()
    _install(monkeypatch, n_frames=2, per_frame=[(None, None), (lms, lms)])
    with pytest.warns(UserWarning, match="no detectable pose"):
        df = keypoints_from_video("clip.mp4", fallback_to_image=True)
    assert df["frame"].nunique() == 1  # only the good frame survives
