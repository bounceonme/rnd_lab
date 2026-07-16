# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Python module serving as a project/extension template.
"""

import importlib.util


# Isaac Sim exposes pxr only after its Python runtime is initialized. Keeping
# registration behind that boundary lets pure hardware modules run on the robot
# without importing the simulator stack.
if importlib.util.find_spec("pxr") is not None:
    from .tasks import *  # noqa: F401, F403
    from .ui_extension_example import *  # noqa: F401, F403
