# Kaggle/Colab project setup — NO PIP INSTALLATIONS.
# Uses the versions already installed in this runtime.
from pathlib import Path
from importlib import metadata
import os, sys, subprocess

REPO_URL = "https://github.com/muhammad-omer-dev/nlp_research_proj.git"
PULL_LATEST = False  # Enable only for a clean clone after pushing fixes to GitHub.
if Path("/kaggle/working").is_dir():
    WORKSPACE = Path("/kaggle/working")
elif Path("/content").is_dir():
    WORKSPACE = Path("/content")
else:
    WORKSPACE = Path.cwd()
REPO_DIR = WORKSPACE / "nlp_research_proj"
ROOT = REPO_DIR / "Urdu_Qwen_Research_Notebooks"

if (REPO_DIR / ".git").is_dir():
    if PULL_LATEST:
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
elif REPO_DIR.exists():
    raise RuntimeError(f"Existing non-git folder: {REPO_DIR}; inspect before removing it")
elif (Path.cwd() / "project_utils.py").is_file():
    ROOT = Path.cwd()
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

if not (ROOT / "project_utils.py").is_file():
    raise FileNotFoundError(f"Project code missing from {ROOT}")

# Check user-confirmed Kaggle versions; NEVER uninstall or upgrade preinstalled dependencies.
EXPECTED = {
    "transformers": "5.16.1",
    "peft": "0.20.0",
    "torch": "2.11.0",  # the Kaggle build has a +cu128 suffix
    "numpy": "2.1.3",
    "scipy": "1.16.3",
    "scikit-learn": "1.6.1",
}
versions = {}
for name, expected in EXPECTED.items():
    try:
        got = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"Missing preinstalled dependency: {name}") from None
    versions[name] = got
    if got.split("+")[0] != expected:
        raise RuntimeError(
            f"This code is checked for {name}={expected}, found {got}. "
            "Do NOT run pip; use the tested Kaggle image or review the API changes."
        )
for name in ("accelerate", "datasets", "pandas", "matplotlib", "pytest"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"Required package missing: {name}. This runtime cannot run the selected notebook.") from None
print("Using preinstalled Kaggle libraries (no installations):", versions)

os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_utils import *
folders()
api = check_training_api()
print("Project ready:", ROOT)
print("Training API ready:", api["transformers"], api["trainer_tokenizer_key"], api["warmup_key"])
