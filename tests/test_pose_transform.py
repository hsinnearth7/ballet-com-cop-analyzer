"""Test the MediaPipe coordinate transform without cv2/mediapipe or a real video."""
from types import SimpleNamespace

from balletcomcop.pose_to_keypoints import landmark_to_xyz


def test_vertical_is_up_positive():
    # MediaPipe y grows downward: a landmark with large +y (low in frame) must map to small z.
    high = SimpleNamespace(x=0.1, y=-0.8, z=0.2)   # near top of frame (y negative)
    low = SimpleNamespace(x=0.1, y=0.5, z=0.2)     # near bottom (y positive)
    _, _, z_high = landmark_to_xyz(high)
    _, _, z_low = landmark_to_xyz(low)
    assert z_high > z_low                          # higher in space -> larger z (up-positive)


def test_columns_are_distinct_no_duplicate_bug():
    lm = SimpleNamespace(x=0.3, y=0.4, z=0.5)
    x, y, z = landmark_to_xyz(lm)
    assert (x, y, z) == (0.3, 0.5, -0.4)           # x=img-x, y=depth(z), z=vert(-y); y != z
