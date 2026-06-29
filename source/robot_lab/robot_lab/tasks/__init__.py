# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The public fork only maintains the non-4bar RND STEP velocity task. Keep the
# upstream robot/task packages out of Gym registration even if local copies exist.
_BLACKLIST_PKGS = [
    "utils",
    "direct",
    "beyondmimic",
    "quadruped",
    "wheeled",
    "others",
    "booster_t1",
    "fftai_gr1t1",
    "fftai_gr1t2",
    "magiclab_magicbot_gen1",
    "magiclab_magicbot_z1",
    "openloong_loong",
    "roboparty_atom01",
    "robotera_xbot",
    "unitree_g1",
    "unitree_h1",
    "rnd_step_4bar",
]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
