"""Fast unit tests for logged CONCORD reference classification metrics."""
import logging
from types import SimpleNamespace
import unittest

from navieco.concord_accuracy import capture_run_concord


class TestAccuracyCapture(unittest.TestCase):
    def test_capture_preserves_original_return_value(self):
        model = SimpleNamespace(navieco_history=[
            {"epoch": 1.0, "train_loss": 4.0, "val_loss": 3.6},
            {"epoch": 2.0, "train_loss": 3.6, "val_loss": 3.0},
        ])

        def fake_run_concord():
            logger = logging.getLogger("concord")
            logger.info("Epoch:   0 | Train accuracy:  0.85 | precision: 0: 0.88")
            logger.info("Epoch:   0 | Val accuracy:  0.89 | precision: 0: 0.90")
            logger.info("Epoch:   1 | Train accuracy:  0.91 | precision: 0: 0.94")
            logger.info("Epoch:   1 | Val accuracy:  0.98 | precision: 0: 0.99")
            return ("joint", model)

        logger = logging.getLogger("concord")
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            actual = capture_run_concord(fake_run_concord)()
        finally:
            logger.setLevel(previous_level)
        self.assertIs(actual[1], model)
        self.assertEqual(len(model.navieco_accuracy_history), 2)
        self.assertEqual(model.navieco_accuracy_history[-1], [2.0, 0.91, 0.98])


if __name__ == "__main__":
    unittest.main()
