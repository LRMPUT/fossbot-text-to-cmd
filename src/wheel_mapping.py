"""Mapping from a recognised action to differential-drive wheel speeds.

The values are normalised motor commands in [-1.0, 1.0]. The same
numeric range was introduced in Lab 2 (low-level wheel control),
so a downstream script could pass these straight to the FOSSBot API.
"""

WHEEL_COMMANDS = {
    "forward":    {"left":  0.5, "right":  0.5},
    "backward":   {"left": -0.5, "right": -0.5},
    "turn_left":  {"left": -0.3, "right":  0.3},
    "turn_right": {"left":  0.3, "right": -0.3},
    "stop":       {"left":  0.0, "right":  0.0},
}

ACTIONS = list(WHEEL_COMMANDS.keys())
