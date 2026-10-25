# Kaggle start guide — Python 3.13 / Transformers 5 / PEFT 0.20

**Do not pip-install any version of NumPy, Torch, Transformers or PEFT.**
Your already-working Kaggle environment reported:

- Python 3.13.15
- Torch 2.11.0+cu128 and CUDA available
- Transformers 5.16.1
- PEFT 0.20.0
- NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1

## Put the new files in GitHub first

Unzip the delivered archive and replace `Urdu_Qwen_Research_Notebooks/` inside
`https://github.com/muhammad-omer-dev/nlp_research_proj.git`.
Be sure to commit **every updated notebook**, `project_utils.py`, tests and `requirements.txt`.
The GitHub repository is not updated automatically by the download.

## Starting in Kaggle

1. Start a fresh Kaggle notebook, enable Internet and NVIDIA GPU if you want to train.
2. Open the updated `00_environment_and_pilot.ipynb` notebook or copy its **first code cell**.
3. It clones the latest GitHub repository and checks versions; it never runs pip.
4. Run the offline tests in notebook 00.
5. Set `RUN_MODEL_SMOKE=True` and run once to validate tokenizer and forward loss.
6. Run `01_preprocess_data.ipynb` to create SIB-200 CSVs (it uses `category`, not `label`).
7. Return to notebook 00; set `RUN_GPU_PILOT=True` for the two-step LoRA/full checks.
8. If both pilots pass, proceed 02, 03, 04, 05, 06, 07.
9. Avoid changing model, data, or library versions between LoRA and full runs.

## If the notebook still uses an old function

Do not patch Kaggle's temporary files. Push all replacements to GitHub, then
restart a fresh Kaggle runtime and clone. Existing clones reuse the old checkout
unless you set `PULL_LATEST=True` with a clean Git working tree.

## Persist outputs

`data/` and `outputs/` are not preserved just by Git clone. Use Kaggle notebook
versions/output artifacts or Drive/another persistent store. Never commit full
model weights to ordinary Git history. If continuing training, restore `outputs/`
with `run_spec.json` and the Trainer `checkpoint-*` folders intact.

## If the environment check fails

Inspect versions and the *first* exception; do not automatically downgrade preinstalled
Kaggle packages. This code targets the exact versions observed above. The initial
code checks only those known versions and requires the other tools to be installed.

## Validation scope

The release passed local pytest, notebook structure checks and syntax checks.
The real Transformers 5.16.1 CUDA/PEFT environment and Qwen model downloads are
not available in the local build environment; Kaggle must validate them with
Notebook 00 before spending GPU time on larger experiments.
