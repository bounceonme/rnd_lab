from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BUFFER_PATH = (
    REPO_ROOT.parent
    / "IsaacLab"
    / "source"
    / "isaaclab"
    / "isaaclab"
    / "utils"
    / "buffers"
    / "circular_buffer.py"
)
SPEC = importlib.util.spec_from_file_location("rnd_test_circular_buffer", BUFFER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load pinned Isaac Lab CircularBuffer from {BUFFER_PATH}.")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CircularBuffer = MODULE.CircularBuffer


class RndObservationHistoryTest(unittest.TestCase):
    def test_first_post_reset_value_prefills_all_four_frames(self):
        history = CircularBuffer(max_len=4, batch_size=2, device="cpu")
        current = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        history.append(current)

        self.assertEqual(history.buffer.shape, (2, 4, 2))
        torch.testing.assert_close(history.buffer, current[:, None, :].expand(-1, 4, -1))

    def test_partial_reset_prefills_only_the_reset_environment(self):
        history = CircularBuffer(max_len=4, batch_size=2, device="cpu")
        history.append(torch.tensor([[1.0], [10.0]]))
        history.append(torch.tensor([[2.0], [20.0]]))
        history.reset(torch.tensor([0]))

        history.append(torch.tensor([[7.0], [30.0]]))

        torch.testing.assert_close(history.buffer[0], torch.full((4, 1), 7.0))
        torch.testing.assert_close(history.buffer[1, :, 0], torch.tensor([10.0, 10.0, 20.0, 30.0]))


if __name__ == "__main__":
    unittest.main()
