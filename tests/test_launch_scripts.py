import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LaunchScriptsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name).resolve()
        self.repo = self.workspace / "repository"
        shutil.copytree(ROOT / "scripts/statistical", self.repo / "scripts/statistical")
        shutil.copytree(ROOT / "scripts/generalization", self.repo / "scripts/generalization")
        shutil.copytree(ROOT / "configs/generalization", self.repo / "configs/generalization")
        self.capture = self.workspace / "commands.jsonl"
        bin_dir = self.workspace / "bin"
        bin_dir.mkdir()
        python_stub = bin_dir / "python"
        python_stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, subprocess, sys\n"
            "if sys.argv[1] == '-c':\n"
            "    sys.exit(subprocess.call([os.environ['REAL_PYTHON'], *sys.argv[1:]]))\n"
            "with open(os.environ['COMMAND_CAPTURE'], 'a') as stream:\n"
            "    stream.write(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}) + '\\n')\n"
        )
        python_stub.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "REAL_PYTHON": sys.executable,
            "COMMAND_CAPTURE": str(self.capture),
            "DATASET_PATH": "dataset/merged_dataset",
        }
        for name in ("PLANS_JSON", "PLAN_WORKERS", "OUTPUT_DIR", "METRICS_JSON"):
            self.env.pop(name, None)

    def run_wrapper(self, path):
        self.capture.write_text("")
        subprocess.run(
            ["bash", str(self.repo / path)],
            cwd=self.workspace,
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        )
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_statistical_wrappers_resolve_paths_and_environment_overrides(self):
        self.env.update(SCORING_MODEL="test/scorer", REFERENCE_MODEL="test/reference")
        for wrapper in sorted((ROOT / "scripts/statistical").glob("run_*.sh")):
            with self.subTest(wrapper=wrapper.name):
                commands = self.run_wrapper(wrapper.relative_to(ROOT))
                self.assertEqual(len(commands), 7)
                for command in commands:
                    self.assertEqual(Path(command["cwd"]), self.repo)
                    args = command["args"]
                    self.assertEqual(args[0], "run_baseline_statistical.py")
                    self.assertTrue((ROOT / args[0]).is_file())
                    self.assertEqual(args[args.index("--scoring-model") + 1], "test/scorer")
                    if args[args.index("--criterion") + 1] in {
                        "fast_detect_gpt", "lastde_pp", "dna_detectllm"
                    }:
                        self.assertEqual(
                            args[args.index("--reference-model") + 1], "test/reference"
                        )

    def test_generalization_wrappers_load_plans_and_separate_output_directories(self):
        outputs = []
        for wrapper in sorted((ROOT / "scripts/generalization").glob("*.sh")):
            with self.subTest(wrapper=wrapper.name):
                commands = self.run_wrapper(wrapper.relative_to(ROOT))
                self.assertEqual(len(commands), 1)
                command = commands[0]
                self.assertEqual(Path(command["cwd"]), self.repo)
                args = command["args"]
                self.assertEqual(args[0], "train_generalization_detector.py")
                self.assertTrue((ROOT / args[0]).is_file())
                plans_path = self.repo / args[args.index("--plans-json") + 1]
                plans = json.loads(plans_path.read_text())
                self.assertEqual(int(args[args.index("--plan-workers") + 1]), len(plans))
                self.assertEqual(args[args.index("--classifier") + 1], "forest")
                self.assertEqual(
                    args[args.index("--normalize-mode") + 1],
                    "raw" if wrapper.stem.endswith("_raw") else "whitespace",
                )
                outputs.append(args[args.index("--output-dir") + 1])
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_plan_coverage_and_training_parser(self):
        sys.path.insert(0, str(ROOT))
        from train_generalization_detector import (
            ensure_unique_plan_names,
            plans_from_json_file,
            resolve_plan,
        )

        folder = ROOT / "configs/generalization"
        one_to_all = json.loads((folder / "one_to_all.json").read_text())
        sources = {plan["train_sources"][0] for plan in one_to_all}
        self.assertEqual(len(one_to_all), 17)
        self.assertEqual(len(sources), 17)
        for plan in one_to_all:
            self.assertEqual(len(plan["train_sources"]), 1)
            self.assertEqual(set(plan["test_sources"]), sources)

        leave_one_out = json.loads((folder / "all_other_to_one.json").read_text())
        self.assertEqual(len(leave_one_out), 17)
        for plan in leave_one_out:
            self.assertEqual(len(plan["test_sources"]), 1)
            self.assertEqual(set(plan["train_sources"]), sources - set(plan["test_sources"]))

        open_to_closed = json.loads((folder / "open_to_closed.json").read_text())
        self.assertEqual(len(open_to_closed), 1)
        plan = open_to_closed[0]
        self.assertEqual(len(plan["train_sources"]), 6)
        self.assertEqual(len(plan["test_sources"]), 11)
        self.assertFalse(set(plan["train_sources"]) & set(plan["test_sources"]))
        self.assertEqual(set(plan["train_sources"]) | set(plan["test_sources"]), sources)

        for path in folder.glob("*.json"):
            plans = plans_from_json_file(str(path))
            ensure_unique_plan_names(plans)
            for plan in plans:
                resolve_plan(
                    plan,
                    human_source="human",
                    train_sources=sorted(sources),
                    test_sources=sorted(sources),
                    train_split="train",
                    test_split="test",
                )


if __name__ == "__main__":
    unittest.main()
