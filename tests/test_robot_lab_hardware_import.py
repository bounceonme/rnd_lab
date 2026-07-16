from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "robot_lab"


class RobotLabHardwareImportTest(unittest.TestCase):
    def test_hardware_runtime_import_does_not_require_isaac_or_pxr(self):
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{PACKAGE_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(PACKAGE_ROOT)
        )

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from robot_lab.hardware import CMP10ARuntimeSource, load_cmp10a_runtime_model; "
                    "model = load_cmp10a_runtime_model('scripts/tools/config/rnd_cmp10a_runtime.json'); "
                    "print(CMP10ARuntimeSource.__name__, model['model_type'])"
                ),
            ],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CMP10ARuntimeSource rnd_cmp10a_policy_observation")


if __name__ == "__main__":
    unittest.main()
