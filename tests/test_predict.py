import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from datasets import Dataset, DatasetDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectors.ascii_detector import ASCIIDetector
from train_merged_ascii_detector import normalize_text


class PredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.folder = Path(cls.tmp.name)
        rows = []
        for i in range(24):
            for source, response in [
                ("human", f"I don't think that is the answer. Line {i}.\r\nHere is why."),
                ("model-a", f"## Answer {i}\r\n\r\nHere’s the result—**yes**."),
                ("model-b", f"### Result {i}\n\n- **First:** compute `x`.\n- Then stop."),
            ]:
                rows.append({"source": source, "response": response, "dataset": "writingprompts"})
        data = Dataset.from_list(rows)
        DatasetDict(train=data, test=data.select(range(30))).save_to_disk(str(cls.folder / "data"))
        cls.models = {}
        for mode in ("raw", "whitespace"):
            model = cls.folder / f"{mode}.pkl"
            subprocess.run([
                sys.executable, str(ROOT / "train_merged_ascii_unified_detector.py"),
                "--dataset-path", str(cls.folder / "data"),
                "--instruct-source", "model-a", "model-b",
                "--normalize-mode", mode, "--classifier", "forest",
                "--n-estimators", "5", "--n-jobs", "1", "--num-workers", "1",
                "--output", str(model),
                "--metrics-json", str(model.with_suffix(".metrics.json")),
            ], cwd=ROOT, capture_output=True, text=True, check=True)
            cls.models[mode] = model

    def run_predict(self, *args):
        return subprocess.run([
            sys.executable, str(ROOT / "predict.py"), *map(str, args)
        ], cwd=self.folder, capture_output=True, text=True)

    def test_training_metadata_and_prediction_preserve_internal_line_endings(self):
        text = " \tHere’s a list:\r\n\r\n  - **one**\r\n\t- two\r\n "
        path = self.folder / "document.txt"
        path.write_bytes(text.encode("utf-8"))
        for mode, model in self.models.items():
            with self.subTest(mode=mode):
                result = self.run_predict("--model", model, "--file", path)
                self.assertEqual(result.returncode, 0, result.stderr)
                output = json.loads(result.stdout)
                detector = ASCIIDetector({"model_path": str(model), "num_workers": 1})
                self.assertAlmostEqual(output["ai_score"], detector.predict(normalize_text(text, mode)))
                self.assertEqual(output["file"], str(path))
                metrics = json.loads(model.with_suffix(".metrics.json").read_text())
                self.assertEqual(metrics["train_source_counts"], {"human": 24, "model-a": 12, "model-b": 12})
                self.assertEqual(set(metrics["per_model_test"]), {"model-a", "model-b"})

    def test_rejects_preprocessing_conflict(self):
        result = self.run_predict("--model", self.models["raw"], "--normalize-mode", "whitespace", "--text", "test")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from the training metadata", result.stderr)

    def test_requires_preprocessing_when_metadata_is_absent(self):
        model = self.folder / "without-metadata.pkl"
        model.write_bytes(self.models["raw"].read_bytes())
        result = self.run_predict("--model", model, "--text", "test")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--normalize-mode", result.stderr)
        result = self.run_predict("--model", model, "--normalize-mode", "raw", "--text", "test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(0 <= json.loads(result.stdout)["ai_score"] <= 1)


if __name__ == "__main__":
    unittest.main()
