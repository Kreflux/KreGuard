import json
import math
import os
import stat
import tempfile
import unittest
from pathlib import Path

from kreguard.adaptive.model import ModelError, OnlineLogReg, load_json, save_json, sigmoid


class OnlineLogRegTests(unittest.TestCase):
    def test_learns_a_separable_problem(self):
        m = OnlineLogReg(dim=64)
        pos, neg = {1: 1.0, 2: 0.5}, {3: 1.0, 4: 0.5}
        for _ in range(20):
            m.update(pos, 1)
            m.update(neg, 0)
        self.assertGreater(m.predict(pos), 0.9)
        self.assertLess(m.predict(neg), 0.1)

    def test_training_is_deterministic(self):
        def run():
            m = OnlineLogReg(dim=64)
            for i in range(50):
                m.update({i % 7: 1.0, (i * 3) % 11: 0.5}, i % 2)
            return m.to_dict()

        self.assertEqual(run(), run())

    def test_weights_are_clipped(self):
        m = OnlineLogReg(dim=64, clip=2.0)
        for _ in range(500):
            m.update({5: 1.0}, 1)
        self.assertLessEqual(max(abs(w) for w in m.w.values()), 2.0)
        self.assertLessEqual(abs(m.bias), 2.0)

    def test_it_follows_a_changing_world(self):
        """The accumulator decays, so a long history cannot freeze the weights.

        After thousands of noisy updates plain AdaGrad (rho=1) has a huge
        accumulator and a tiny step size. The decaying accumulator stays
        responsive and flips on the new regime several times faster.
        """
        import random

        def updates_to_flip(rho):
            m = OnlineLogReg(dim=64, rho=rho)
            f, rng = {9: 1.0}, random.Random(3)
            for _ in range(3000):
                m.update(f, 1 if rng.random() < 0.9 else 0)
            self.assertGreater(m.predict(f), 0.8)
            for i in range(1, 2000):
                m.update(f, 0)
                if m.predict(f) < 0.5:
                    return i
            return 2000

        adaptive, frozen = updates_to_flip(0.99), updates_to_flip(1.0)
        self.assertLessEqual(adaptive, 30)
        self.assertGreater(frozen, 2 * adaptive)

    def test_pruning_keeps_the_strongest_weights(self):
        m = OnlineLogReg(dim=1 << 12, max_features=50)
        m.update({1: 1.0}, 1)
        for _ in range(5):
            m.update({1: 1.0}, 1)
        for i in range(100, 300):
            m.update({i: 0.01}, 1)
        self.assertLessEqual(len(m.w), 50)
        self.assertIn(1, m.w)

    def test_sigmoid_is_stable(self):
        self.assertEqual(sigmoid(1e6), 1.0)
        self.assertLess(sigmoid(-1e6), 1e-20)
        self.assertTrue(math.isfinite(sigmoid(-1e6)))
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_bad_hyperparameters_rejected(self):
        for kw in ({"dim": 100}, {"lr": 0}, {"rho": 0}, {"clip": -1}):
            with self.assertRaises(ValueError):
                OnlineLogReg(**kw)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "m.json"
        self.model = OnlineLogReg(dim=64)
        for i in range(30):
            self.model.update({i % 5: 1.0}, i % 2)

    def test_roundtrip_preserves_predictions(self):
        save_json(self.path, "test", self.model.to_dict())
        loaded = OnlineLogReg.from_dict(load_json(self.path, "test"))
        for i in range(5):
            self.assertAlmostEqual(loaded.predict({i: 1.0}), self.model.predict({i: 1.0}), places=6)

    def test_file_is_owner_only_and_leaves_no_temp_files(self):
        save_json(self.path, "test", self.model.to_dict())
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        self.assertEqual([p.name for p in Path(self.tmp.name).iterdir()], ["m.json"])

    def test_edited_file_fails_its_integrity_check(self):
        save_json(self.path, "test", self.model.to_dict())
        doc = json.loads(self.path.read_text())
        doc["payload"]["bias"] = 0.5
        self.path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ModelError, "integrity"):
            load_json(self.path, "test")

    def test_wrong_kind_garbage_and_constants_are_rejected(self):
        save_json(self.path, "test", self.model.to_dict())
        with self.assertRaises(ModelError):
            load_json(self.path, "other")
        self.path.write_text("{nope")
        with self.assertRaises(ModelError):
            load_json(self.path, "test")
        self.path.write_text('{"format":"kreguard-test","version":1,"sha256":"x","payload":{"a":NaN}}')
        with self.assertRaises(ModelError):
            load_json(self.path, "test")
        with self.assertRaises(ModelError):
            load_json(Path(self.tmp.name) / "missing.json", "test")

    def test_valid_checksum_does_not_excuse_bad_numbers(self):
        """An attacker who recomputes the checksum still cannot load nonsense."""
        for field, bad in (("bias", 1e9), ("lr", -1), ("dim", 100)):
            payload = self.model.to_dict()
            payload[field] = bad
            save_json(self.path, "test", payload)
            with self.assertRaises(ModelError, msg=field):
                OnlineLogReg.from_dict(load_json(self.path, "test"))
        payload = self.model.to_dict()
        payload["w"][0] = 1e6
        with self.assertRaises(ModelError):
            OnlineLogReg.from_dict(payload)
        payload = self.model.to_dict()
        payload["idx"][0] = 10_000
        with self.assertRaises(ModelError):
            OnlineLogReg.from_dict(payload)
        payload = self.model.to_dict()
        payload["w"] = payload["w"][:-1]
        with self.assertRaises(ModelError):
            OnlineLogReg.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
