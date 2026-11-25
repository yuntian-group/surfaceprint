from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_merged_ascii_detector import resolve_ascii_detector_class


def load_analysis_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FeatureAnalysisTest(unittest.TestCase):
    def test_saved_forests_use_metadata_for_registry_and_ablations(self):
        forest_analysis = load_analysis_script("analyze_ascii_forest_features")
        general_analysis = forest_analysis.aaf
        texts = [
            "a plain reply. it has no headings",
            "## Answer\n\n**Yes.** Here’s why:\n- First\n- Second\n",
            "i think the answer is 4, not 5",
            "### Result\n\nThe result is \\[x = 4\\].\n",
            "try print(x) and check the value",
            "**Solution**\n\n```python\nprint(x)\n```\n",
            "that isn't what i meant...",
            "## Explanation\n\nIt’s clear—see **the example**.\n",
        ]
        labels = [0, 1] * 4
        cases = [
            ("full", []), ("full", ["E"]),
            ("binary", []), ("binary", ["E"]),
            ("basic", []), ("basic", ["char"]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name, disabled in cases:
                with self.subTest(registry=name, disabled=disabled):
                    detector = resolve_ascii_detector_class(name)({
                        "model": RandomForestClassifier(n_estimators=4, max_depth=3, random_state=7),
                        "num_workers": 1,
                        "fit_show_progress": False,
                        "disabled_feature_groups": disabled,
                    }).fit(texts, labels)
                    model_path = folder / "detector.pkl"
                    detector.save(model_path)
                    model_path.with_suffix(".metrics.json").write_text(json.dumps({
                        "ascii_detector": name,
                        "disable_feature_group": disabled,
                    }))

                    for module in (forest_analysis, general_analysis):
                        output = folder / f"{module.__name__}.json"
                        argv = [
                            module.__file__, "--model", str(model_path),
                            "--show-zero", "--json-output", str(output),
                        ]
                        if module is forest_analysis:
                            argv += ["--split-usage", "--per-tree", "--tree-index", "0"]
                        stdout = io.StringIO()
                        with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                            module.main()
                        result = json.loads(output.read_text())
                        self.assertEqual(result["summary"]["ascii_detector"], name)
                        self.assertEqual(result["summary"]["disabled_feature_groups"], disabled)
                        self.assertEqual(result["summary"]["registered_features"], detector.feature_dim)
                        features = sorted(result["feature_importances"], key=lambda row: row["index"])
                        self.assertEqual([row["feature"] for row in features], detector.feature_names)
                        np.testing.assert_allclose(
                            [row["importance"] for row in features], detector.model.feature_importances_
                        )
                        self.assertTrue(all(row["group_label"] != "unknown" for row in features))
                        self.assertFalse(set(disabled) & {row["group"] for row in features})
                        self.assertIn(f"ascii_detector: {name}", stdout.getvalue())
                        self.assertIn("Group importances", stdout.getvalue())
                        if module is forest_analysis:
                            self.assertEqual(result["summary"]["n_estimators"], 4)
                            self.assertEqual(len(result["per_tree"]), 4)
                            self.assertTrue(result["split_usage"])
                            self.assertTrue(all(row["group_label"] != "unknown" for row in result["split_usage"]))
                            self.assertIn("Decision tree #0", stdout.getvalue())
                            self.assertIn("Split usage across forest", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
