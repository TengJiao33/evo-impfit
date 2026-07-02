import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evo_impfit.data import make_scenario, scenario_from_csv
from evo_impfit.experiments import parse_variant_names, summarize_runs, RunSummary
from evo_impfit.features import FeatureConfig, evaluate_objectives, weighted_mse
from evo_impfit.optimizer import run_fitting


class SmokeTests(unittest.TestCase):
    def test_true_case1_has_low_loss(self):
        scenario = make_scenario("rlc_case1")
        cfg = FeatureConfig()
        target_fit = scenario.model(scenario.frequencies_hz, scenario.true_values)
        self.assertLess(weighted_mse(scenario.target_z, target_fit, cfg.alpha), 1e-18)

    def test_feature_objectives_are_finite(self):
        scenario = make_scenario("rlc_case1", n_points=48)
        cfg = FeatureConfig(dtw_points=24)
        rng = np.random.default_rng(3)
        random_values = np.array(
            [
                10 ** (np.log10(spec.lower) + rng.random() * (np.log10(spec.upper) - np.log10(spec.lower)))
                for spec in scenario.specs
            ]
        )
        fit = scenario.model(scenario.frequencies_hz, random_values)
        values = evaluate_objectives(
            ("wsmse", "ptc", "dtw", "lfp", "ss", "lmep"),
            scenario.frequencies_hz,
            scenario.target_z,
            fit,
            cfg,
        )
        self.assertTrue(all(np.isfinite(value) for value in values.values()))

    def test_optimizer_runs(self):
        scenario = make_scenario("motor_synthetic", n_points=36)
        result = run_fitting(
            scenario,
            generations=2,
            population_size=12,
            seed=5,
            cfg=FeatureConfig(dtw_points=18),
        )
        self.assertTrue(np.isfinite(result.best_loss))
        self.assertEqual(len(result.history), 3)

    def test_experiment_summary(self):
        self.assertEqual(parse_variant_names("wbeif,femeif_light"), ("wbeif", "femeif_light"))
        rows = [
            RunSummary("a", "femeif", ("wsmse",), 0, 4.0, 1.0, 2, 10),
            RunSummary("a", "femeif", ("wsmse",), 1, 2.0, 3.0, 2, 10),
            RunSummary("b", "wbeif", ("wsmse",), 0, 5.0, 2.0, 2, 10),
        ]
        summary = summarize_runs(rows)
        by_name = {row["variant"]: row for row in summary}
        self.assertEqual(by_name["a"]["runs"], 2)
        self.assertAlmostEqual(by_name["a"]["mean_best_wsmse"], 3.0)

    def test_csv_scenario_loader(self):
        scenario = make_scenario("rlc_case1", n_points=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frequency_hz", "real", "imag"])
                for freq, z_value in zip(scenario.frequencies_hz, scenario.target_z):
                    writer.writerow([freq, z_value.real, z_value.imag])
            loaded = scenario_from_csv(path, "rlc6")
        self.assertEqual(len(loaded.frequencies_hz), 8)
        self.assertEqual(len(loaded.specs), 18)


if __name__ == "__main__":
    unittest.main()
