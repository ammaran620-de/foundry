"""Tests for benchmark/regression_gate.py, run through its command line so exit codes are checked.

The gate returns 0 (within tolerance), 1 (regression) or 2 (the inputs cannot be compared). Candidates here are
synthetic files built for each case, plus the recorded results in benchmark/results/. Run with:

    python -m unittest discover -s tests -v

Only the standard library is used, so the same command runs on every CI platform.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "benchmark" / "regression_gate.py"
RESULTS = ROOT / "benchmark" / "results"
DEPLOYMENT_ACCURACY = RESULTS / "deployment_accuracy.json"
SUSTAINED_LATENCY = RESULTS / "sustained_int8_robust.json"


def accuracy(map_value, **extra):
    return {"mAP50_95": map_value, **extra}


def latency(p95, device="Test CPU"):
    return {"results": {"CPU": {"device_name": device, "overall": {"p95_ms": p95}}}}


class GateCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)

    def write(self, name, data, encoding="utf-8"):
        path = self.dir / name
        text = data if isinstance(data, str) else json.dumps(data)   # json.dumps writes NaN and Infinity as-is
        path.write_text(text, encoding=encoding)
        return path

    def run_gate(self, ref_acc, cand_acc, ref_lat, cand_lat, *extra):
        proc = subprocess.run(
            [sys.executable, str(GATE),
             "--reference-accuracy", str(ref_acc), "--candidate-accuracy", str(cand_acc),
             "--reference-latency", str(ref_lat), "--candidate-latency", str(cand_lat), *extra],
            capture_output=True, text=True, cwd=self.dir,
        )
        return proc.returncode, proc.stdout + proc.stderr

    def compare(self, ref_map, cand_map, ref_p95, cand_p95, *extra):
        return self.run_gate(
            self.write("ra.json", accuracy(ref_map)), self.write("ca.json", accuracy(cand_map)),
            self.write("rl.json", latency(ref_p95)), self.write("cl.json", latency(cand_p95)), *extra,
        )


class PassingAndFailingCandidates(GateCase):
    def test_better_candidate_passes(self):
        code, out = self.compare(0.40, 0.41, 50.0, 45.0)
        self.assertEqual(code, 0, out)
        self.assertIn("RESULT: PASS", out)

    def test_small_loss_inside_tolerance_passes(self):
        code, out = self.compare(0.40, 0.395, 50.0, 52.5)     # -0.5 points, +5%
        self.assertEqual(code, 0, out)

    def test_accuracy_loss_over_tolerance_fails(self):
        code, out = self.compare(0.40, 0.38, 50.0, 50.0)      # -2 points
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL: accuracy regression", out)
        self.assertIn("PASS: latency regression", out)
        self.assertIn("RESULT: FAIL", out)

    def test_latency_growth_over_tolerance_fails(self):
        code, out = self.compare(0.40, 0.40, 50.0, 60.0)      # +20%
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL: latency regression", out)
        self.assertIn("PASS: accuracy regression", out)

    def test_both_failing_fails(self):
        code, out = self.compare(0.40, 0.30, 50.0, 80.0)
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL: accuracy", out)
        self.assertIn("FAIL: latency", out)

    def test_exactly_at_tolerance_passes_and_just_over_fails(self):
        self.assertEqual(self.compare(0.40, 0.39, 50.0, 55.0)[0], 0)       # 1.0 point loss, 10% growth
        self.assertEqual(self.compare(0.40, 0.3899, 50.0, 50.0)[0], 1)
        self.assertEqual(self.compare(0.40, 0.40, 50.0, 55.01)[0], 1)

    def test_tolerances_are_configurable(self):
        self.assertEqual(self.compare(0.40, 0.38, 50.0, 50.0)[0], 1)
        self.assertEqual(self.compare(0.40, 0.38, 50.0, 50.0, "--accuracy-tolerance-map-points", "2")[0], 0)
        self.assertEqual(self.compare(0.40, 0.40, 50.0, 60.0, "--latency-tolerance-percent", "25")[0], 0)

    def test_invalid_tolerances_are_rejected(self):
        for bad in ("-1", "nan", "inf", "abc"):
            code, _ = self.compare(0.40, 0.40, 50.0, 50.0, "--accuracy-tolerance-map-points", bad)
            self.assertEqual(code, 2, bad)        # argparse error


class InputsThatCannotBeCompared(GateCase):
    def assertInputError(self, result, fragment):
        code, out = result
        self.assertEqual(code, 2, out)
        self.assertIn("REGRESSION GATE ERROR", out)
        self.assertIn(fragment, out)
        self.assertNotIn("Traceback", out)

    def test_nan_and_infinite_values_are_errors_not_passes(self):
        self.assertInputError(self.compare(0.40, float("nan"), 50.0, 50.0), "not finite")
        self.assertInputError(self.compare(0.40, 0.40, 50.0, float("inf")), "not finite")
        self.assertInputError(self.compare(float("nan"), 0.40, 50.0, 50.0), "not finite")

    def test_non_numbers_are_errors(self):
        self.assertInputError(self.compare(0.40, "0.40", 50.0, 50.0), "not a number")
        self.assertInputError(self.compare(0.40, True, 50.0, 50.0), "not a number")

    def test_map_given_as_a_percentage_is_an_error(self):
        self.assertInputError(self.compare(40.5, 40.5, 50.0, 50.0), "fraction, not a percentage")

    def test_zero_or_negative_latency_is_an_error(self):
        self.assertInputError(self.compare(0.40, 0.40, 0.0, 50.0), "must be positive")
        self.assertInputError(self.compare(0.40, 0.40, 50.0, -3.0), "must be positive")

    def test_missing_keys_and_files(self):
        empty = self.write("empty.json", {})
        ok_acc, ok_lat = self.write("a.json", accuracy(0.4)), self.write("l.json", latency(50.0))
        self.assertInputError(self.run_gate(empty, ok_acc, ok_lat, ok_lat), "Could not find mAP50_95")
        self.assertInputError(self.run_gate(ok_acc, ok_acc, empty, ok_lat), "Could not find CPU p95")
        self.assertInputError(self.run_gate(self.dir / "nope.json", ok_acc, ok_lat, ok_lat), "not found")

    def test_invalid_json_and_non_object_json(self):
        ok_acc, ok_lat = self.write("a.json", accuracy(0.4)), self.write("l.json", latency(50.0))
        self.assertEqual(self.run_gate(self.write("bad.json", "{not json"), ok_acc, ok_lat, ok_lat)[0], 2)
        self.assertInputError(self.run_gate(self.write("list.json", "[1, 2]"), ok_acc, ok_lat, ok_lat), "Expected a JSON object")

    def test_byte_order_mark_is_accepted(self):
        bom = self.write("bom.json", accuracy(0.40), encoding="utf-8-sig")
        lat = self.write("l.json", latency(50.0))
        self.assertEqual(self.run_gate(bom, bom, lat, lat)[0], 0)

    def test_different_image_sizes_are_not_comparable(self):
        ra, ca = self.write("ra.json", accuracy(0.4, image_size=640)), self.write("ca.json", accuracy(0.4, image_size=320))
        lat = self.write("l.json", latency(50.0))
        self.assertInputError(self.run_gate(ra, ca, lat, lat), "different image sizes")

    def test_latency_from_different_cpus_needs_an_explicit_override(self):
        acc = self.write("a.json", accuracy(0.4))
        rl, cl = self.write("rl.json", latency(50.0, "CPU A")), self.write("cl.json", latency(50.0, "CPU B"))
        self.assertInputError(self.run_gate(acc, acc, rl, cl), "different CPUs")
        self.assertEqual(self.run_gate(acc, acc, rl, cl, "--allow-device-mismatch")[0], 0)


class DeploymentAccuracyFiles(GateCase):
    """deployment_accuracy.json holds the FP32 reference and each deployed runtime in one file. An earlier
    version of the gate read the FP32 reference block for both sides, so it could never see a deployment
    lose accuracy."""

    def multi(self, **runs):
        return self.write("multi.json", {"pytorch_fp32_reference": {"mAP50_95": 0.40},
                                         "results": {name: {"mAP50_95": value} for name, value in runs.items()}})

    def test_a_file_with_several_runs_must_name_the_one_to_compare(self):
        multi, lat = self.multi(onnx=0.30), self.write("l.json", latency(50.0))
        code, out = self.run_gate(multi, multi, lat, lat)
        self.assertEqual(code, 2, out)
        self.assertIn("pytorch_fp32_reference, onnx", out)
        self.assertIn("--candidate-deployment", out)

    def test_the_named_candidate_is_the_one_measured(self):
        multi, lat = self.multi(onnx=0.30, good=0.40), self.write("l.json", latency(50.0))
        base = ["--reference-deployment", "pytorch_fp32_reference"]
        self.assertEqual(self.run_gate(multi, multi, lat, lat, *base, "--candidate-deployment", "onnx")[0], 1)
        self.assertEqual(self.run_gate(multi, multi, lat, lat, *base, "--candidate-deployment", "good")[0], 0)
        self.assertEqual(self.run_gate(multi, multi, lat, lat, *base, "--candidate-deployment", "pytorch_fp32_reference")[0], 0)

    def test_unknown_deployment_is_an_error(self):
        multi, lat = self.multi(onnx=0.30), self.write("l.json", latency(50.0))
        code, out = self.run_gate(multi, multi, lat, lat, "--reference-deployment", "pytorch_fp32_reference",
                                  "--candidate-deployment", "tensorrt")
        self.assertEqual(code, 2, out)
        self.assertIn("no deployment 'tensorrt'", out)

    def test_single_run_file_in_the_baseline_layout_is_read(self):
        base = self.write("base.json", {"metrics": {"map50_95": 0.405293}})
        lat = self.write("l.json", latency(50.0))
        self.assertEqual(self.run_gate(base, base, lat, lat)[0], 0)


class RecordedResults(GateCase):
    """The committed results in benchmark/results/. Numbers are read from the files, not copied here."""

    def recorded(self):
        data = json.loads(DEPLOYMENT_ACCURACY.read_text(encoding="utf-8-sig"))
        return {"pytorch_fp32_reference": data["pytorch_fp32_reference"]["mAP50_95"],
                **{k: v["mAP50_95"] for k, v in data["results"].items()}}

    def gate(self, reference, candidate, *extra):
        return self.run_gate(DEPLOYMENT_ACCURACY, DEPLOYMENT_ACCURACY, SUSTAINED_LATENCY, SUSTAINED_LATENCY,
                             "--reference-deployment", reference, "--candidate-deployment", candidate, *extra)

    def test_self_comparison_passes_for_every_recorded_run(self):
        """What CI runs. Passes by construction, so it checks the script and the file layout, not a regression."""
        for name in self.recorded():
            code, out = self.gate(name, name)
            self.assertEqual(code, 0, f"{name}\n{out}")
            self.assertIn("+0.000 points", out)

    def test_recorded_runtime_accuracy_is_below_the_fp32_reference_and_fails_the_default_gate(self):
        """Recorded fact, not a defect of the gate: ONNX Runtime CPU and OpenVINO FP16 score well below the
        PyTorch FP32 reference on the same 1,000-image subset. The gate reports it; the cause is not established."""
        runs = self.recorded()
        for candidate in ("onnxruntime_cpu", "openvino_fp16_cpu"):
            code, out = self.gate("pytorch_fp32_reference", candidate)
            loss = (runs["pytorch_fp32_reference"] - runs[candidate]) * 100
            self.assertGreater(loss, 1.0)
            self.assertEqual(code, 1, out)
            self.assertIn(f"{-loss:+.3f} points", out)
            self.assertEqual(self.gate("pytorch_fp32_reference", candidate, "--accuracy-tolerance-map-points",
                                       str(round(loss + 0.01, 2)))[0], 0)

    def test_degraded_copy_of_the_recorded_latency_fails(self):
        data = json.loads(SUSTAINED_LATENCY.read_text(encoding="utf-8-sig"))
        data["results"]["CPU"]["overall"]["p95_ms"] *= 1.2
        slower = self.write("slower.json", data)
        code, out = self.run_gate(DEPLOYMENT_ACCURACY, DEPLOYMENT_ACCURACY, SUSTAINED_LATENCY, slower,
                                  "--reference-deployment", "onnxruntime_cpu", "--candidate-deployment", "onnxruntime_cpu")
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL: latency regression", out)


if __name__ == "__main__":
    unittest.main()
