# Urdu Small-LLM Research: LoRA vs Full Fine-Tuning

**Project:** Efficient Urdu Adaptation of a Small LLM: LoRA vs Full Fine-Tuning, Data Efficiency and English Retention

**Main model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), Apache 2.0 open weights.  
**Primary dataset:** [Davlan/SIB-200](https://huggingface.co/datasets/Davlan/sib200), English `eng_Latn` and Urdu `urd_Arab`.  
**Published benchmark:** [Generalists vs Specialists, Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.426/), [official code](https://github.com/sameearif/Generalists-vs-Specialists).  
**Optional benchmark training source:** [Urdu News Headline Dataset](https://github.com/mwaseemrandhawa/Urdu-News-Headline-Dataset).

**Status: implementation supplied; real GPU training/results NOT run here.** The scripts were checked locally with automated CPU tests and the first notebook was executed locally, but this environment does not contain the Hugging Face training packages, CUDA GPU or downloadable Qwen weights. All accuracy tables are intentionally blank until you run the experiments.

## 1. Notebook order

| Order | Notebook | What it does | GPU? |
|---|---|---|---|
| 0 | `00_environment_and_pilot.ipynb` | System checks, automated tests, optional tokenizer/model and 2-step training pilot | Optional, needed for pilot |
| 1 | `01_preprocess_data.ipynb` | Download Urdu+English SIB-200, preserve splits, validate IDs/labels, sample few-shot train | No |
| 2 | `02_zero_fewshot_baselines.ipynb` | Run untouched model, zero-shot and in-context few-shot | Recommended |
| 3 | `03_train_lora.ipynb` | Fine-tune trainable adapters, rank 8/16, log resource use, save adapters | Yes |
| 4 | `04_train_full.ipynb` | Full fine-tune same model, batches, loss and optimizer family | Yes |
| 5 | `05_inference.ipynb` | Predict Urdu/English validation and then locked test; automatic inference resume | Recommended |
| 6 | `06_evaluate.ipynb` | Reject incomplete or mismatched predictions; accuracy, macro-F1 and invalid rate | No |
| 7 | `07_results_analysis.ipynb` | Comparison tables, errors, plots, measured training time | No |
| 8 | `08_paper_benchmark_audit.ipynb` | Inspect the LUMS paper's 8-class topic test set, never train on it | No |
| 9 | `09_optional_english_anchor.ipynb` | Produce a common English-trained starting checkpoint for forgetting study | Yes |
| 10 | `10_runtime_planner.ipynb` | Estimates actual workload from measured pilot steps | No |
| 11 | `11_paper8_original_headlines.ipynb` | Build separate eight-class training + validation from upstream headlines; exclude published test data | No |
| 12 | `12_uncertainty_and_retention.ipynb` | Paired bootstrap differences, English retention, confusion matrix | No |

### Do this first: load code from GitHub

The notebooks now **clone this repository automatically in their first code cell**:

```bash
git clone https://github.com/muhammad-omer-dev/nlp_research_proj.git
```

The GitHub repository contains the project inside a nested folder:

```text
nlp_research_proj/
  Urdu_Qwen_Research_Notebooks/
    project_utils.py
    config.json
    requirements.txt
    notebooks/
    tests/
```

**Kaggle:** Enable **Internet** in Notebook Settings. The first code cell clones into
`/kaggle/working/nlp_research_proj/Urdu_Qwen_Research_Notebooks`.
Select a GPU such as T4 for the training notebooks. Run the first code cell at the
start of each notebook. You do **not** need to upload the ZIP or Python files again.

**Google Colab:** Enable GPU from Runtime → Change runtime type; the first code cell
clones into `/content/nlp_research_proj/Urdu_Qwen_Research_Notebooks`.
You do **not** need to upload the project files again.

- An existing cloned repository is reused instead of cloned twice.
- If you push changed source code to GitHub **during the same live session**, set
  `PULL_LATEST = True` in the first code cell to fast-forward an existing clean clone.
- Restarting a Kaggle/Colab runtime clears its temporary filesystem. The next run
  clones again automatically, provided Internet is enabled.
- Cloning GitHub does **not** restore previous GPU checkpoints, training data files,
  saved predictions, or metrics: `data/` and `outputs/` are runtime artifacts. Copy
  them to Google Drive, Kaggle Dataset/Outputs, or another persistent location after
  a session. Never push large model checkpoints to normal GitHub Git history.
- A Git clone requires a public repository or suitable authentication. The supplied
  GitHub repository is currently public.

### Recommended notebook order

1. Run notebook **00**, then notebook **01** to prepare the dataset.
2. In notebook 00, set `INSTALL_PACKAGES=True` once if missing packages are found;
   restart the kernel after installation.
3. After preprocessing, run the optional GPU pilot in notebook 00; test LoRA first,
   then full tuning to check memory and timing.
4. Then run **02 → 03 → 04 → 05 → 06 → 07**. Other notebooks are optional.
5. Set `SPLIT='test'` only after finishing validation and model selection.

**Notebook setup example (already included as the first code cell):**

```python
from pathlib import Path
import subprocess

repo_dir = Path('/kaggle/working/nlp_research_proj')  # use /content on Colab
if not repo_dir.exists():
    subprocess.check_call([
        'git', 'clone', '--depth', '1',
        'https://github.com/muhammad-omer-dev/nlp_research_proj.git',
        str(repo_dir)
    ])
ROOT = repo_dir / 'Urdu_Qwen_Research_Notebooks'
```

## 2. Fair experimental design

**One model** is used for all main training comparisons: Qwen2.5-0.5B-Instruct. All runs use:

- Same dataset and official train/validation/test split.
- Same 7 labels, tokenization/chat template, completion-only loss, maximum token length, effective batch size, and greedy decoding.
- Same optimizer family (**Adafactor**) for LoRA and full tuning. This choice reduces optimizer memory so full tuning is more feasible on 16 GB GPUs. Learning rates differ because fine-tuning methods often need different rates; adjust fairly using validation only.
- Same seeds and identical nested stratified training subsets for comparable conditions.
- Same evaluation with **accuracy, macro-F1, invalid-label count**, training wall time, peak PyTorch GPU-allocated memory and trainable parameter counts.

**Starting run:** 10 examples per class, seed 42, rank 8, 2 epochs. Later add 25 and 50 per class; rank 16; seeds 123 and 456. The notebooks will refuse sample sizes if any class lacks enough examples. Do not oversample silently.

**English forgetting experiment:** The cleanest design first trains a common English checkpoint in notebook 09, merges that English LoRA adapter, then runs Urdu LoRA and Urdu full from that SAME checkpoint. Compare English macro-F1 **before** and **after** Urdu training. Using an unadapted Qwen English baseline is a weaker retention check, not automatically evidence of catastrophic forgetting.

## 3. Published benchmark: important mismatch

- SIB-200 has **seven** labels: science/technology, travel, politics, sports, health, entertainment, geography.
- The LUMS EMNLP 2024 paper's **topic classification** benchmark has **eight** labels: business, entertainment, health, politics, science, sports, other, world.
- The paper reported **84.74 macro-F1** for its LoRA-trained Llama-3-8B on the paper's 1,000 topic test examples.
- **Never compare a SIB-200 macro-F1 directly with 84.74**; they are different classification tasks.
- The paper repository provides `datasets/topic-classification.csv` (columns `input,label`) for held-out **TEST evaluation**, not for training.
- If you want a direct paper-score comparison, run notebook 08 and then 11. Notebook 11 builds independent training/validation from the original headlines repository, discards exact test matches, and preserves the paper's original test.
- The upstream source includes `Weird.xlsx`; mapping it to the paper's `other` label must be checked by a human before import. The original headline repo does not clearly advertise a redistribution license; do not redistribute those spreadsheets with your work.
- Even on identical test data, model sizes and original training samples differ; explain these limitations honestly.

## 4. GPU cost, runtime and recovery

The software and public model can be used without a paid API. **Free GPU availability is not guaranteed**. Full tuning a 0.5B model on a 16 GB T4 is plausible with these conservative settings, but still requires a real pilot. If it runs out of memory, reduce `max_length` from 256 to 192 or 128, keep `per_device_train_batch_size=1`, and restart. This changes your experimental settings: rerun both methods with the same sequence limit.

Training uses gradient checkpointing and **step-based saved checkpoints every 10 optimizer steps** for regular runs. On re-opening a notebook from the same folder, a partially completed run can resume only when a compatible saved checkpoint and identical `run_spec.json` exist. Completed runs refuse to overwrite. Pilot runs save only on success and cannot resume mid-run.

Inference writes every prediction to `*.progress.jsonl` and resumes after interruption, while checking run metadata and source hashes; a final CSV is written only after all requested examples are covered.

Training time is **measured**, not invented. Notebook 00 runs 2-step pilots, 10 uses the measured average time per step and a stated effective batch size to make an approximate quota plan. Warmup/download/checkpoint overhead means the 2-step estimate is weak; use 10–20 steps for a steadier measurement. Check `outputs/metrics/train_runs.jsonl` for actual completed-run times.

Training output location: `outputs/checkpoints/<run_id>/model` (full model or LoRA adapter) plus `complete.json`. Intermediate `checkpoint-*` directories are for trainer resumption. Save persistent copies **before a free GPU session ends**.

## 5. How to read the results

- `outputs/metrics/train_runs.jsonl`: completed training configurations, wall time, peak GPU allocated memory (not the whole device footprint), trainable parameters, truncation counts, source hashes.
- `outputs/predictions/*.csv`: reference, raw generated text, normalized predicted label, validity, per-item generation time.
- `outputs/predictions/*.meta.json`: model, task, language, evaluation split, dataset SHA256, decoding configuration.
- `outputs/metrics/*_metrics.json`: evaluated accuracy, macro-F1, per-class report and confusion matrix.
- `outputs/plots/`: report-ready PNG figures.

**Uncertain or invalid predictions:** The evaluator counts outputs outside the exact allowed labels as incorrect. It never chooses a label merely because its word occurs inside a longer sentence.

## 6. Important limitations / honesty checklist

- A model may produce Urdu descriptions instead of one English label. Invalid rate matters.
- Urdu SIB-200 is small and topic datasets may be imbalanced. Always report category-wise F1 and test sample size.
- If long texts exceed the configured token budget, the training encoder keeps the entire target and shortens source text only; it records how many examples were shortened. Do not silently omit these numbers.
- Automatic scores for generated text are not proof of general Urdu fluency.
- Test data must not be used for selecting rank, epochs, learning rates or checkpoint.
- Never claim to beat the published Llama unless using the **same eight-category paper test set** and macro-F1 procedure. Reproduction on different training samples remains a related but not identical comparison.
- Software-version differences on future GPU images may need small compatibility adjustments. The current `requirements.txt` pins a consistent widely-used Transformers/PEFT stack but this project has **not** executed a full GPU run in this environment.

## 7. Local quality checks that WERE completed

- `pytest -q` → all local unit tests passed.
- All `.ipynb` files validated as Jupyter Notebook v4 and all code cells parsed as Python.
- Notebook 00 executed end to end using the local CPU kernel (GPU optional cells disabled).
- Qwen model/LoRA/full runs and actual Urdu dataset downloads have **not** been run in this environment. The notebooks are designed for your Kaggle/Colab GPU session.

### Source links

1. Qwen official card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
2. SIB-200 official card: https://huggingface.co/datasets/Davlan/sib200
3. PEFT Transformers docs: https://huggingface.co/docs/transformers/main/peft
4. Hugging Face Trainer docs: https://huggingface.co/docs/transformers/main_classes/trainer
5. Paper: https://aclanthology.org/2024.findings-emnlp.426/
6. Paper code: https://github.com/sameearif/Generalists-vs-Specialists
7. Original 8-class source: https://github.com/mwaseemrandhawa/Urdu-News-Headline-Dataset
