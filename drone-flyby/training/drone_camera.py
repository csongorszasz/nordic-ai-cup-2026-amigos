"""The drone camera's geometry, shared by render_city.py and synth_dataset.py.

Kept free of OpenGL imports so the dataset build runs where pyrender is not installed (IDUN).
"""

import numpy as np

WIDTH, HEIGHT = 3840, 2160
HFOV = 68.0          # degrees, estimated from the recorded frames
# The drone camera is pitched forward: tall objects in the supplied frames lean away from
# a point near the bottom edge, (1920, ~2100) in the 4K frame (fitted tilt and lean of the
# towers and launchers in models/*/match.json agree with it to within the 45 deg fit grid).
NADIR_Y = 2100
FOCAL_PX = WIDTH / 2 / np.tan(np.radians(HFOV / 2))
PITCH_DEG = float(np.degrees(np.arctan((NADIR_Y - HEIGHT / 2) / FOCAL_PX)))  # ~19.7
METRES_PER_PIXEL = 0.21


def lean_at(px, py):
    """(tilt, lean) of a tall object at 4K frame pixel (px, py), in render_models' convention.

    Tilt: how far off vertical the camera sees it; lean: image direction its top leans
    toward, degrees clockwise from up. Both follow from the nadir point (WIDTH/2, NADIR_Y).
    """
    dx, dy = px - WIDTH / 2, py - NADIR_Y
    return float(np.degrees(np.arctan(np.hypot(dx, dy) / FOCAL_PX))), float(np.degrees(np.arctan2(dx, -dy)) % 360)
