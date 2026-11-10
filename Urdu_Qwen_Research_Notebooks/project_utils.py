"""Utilities for the Urdu LoRA vs full fine-tuning project.

Kaggle/Colab, Transformers 5.x + PEFT. Data and basic tests can run without a GPU.
Update: validate every full epoch, use early stopping, and restore the best model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = [
    "science/technology", "travel", "politics", "sports", "health",
    "entertainment", "geography",
]
PAPER_LABELS = [
    "business", "entertainment", "health", "politics", "science",
    "sports", "other", "world",
]


def folders():
    for name in ["data", "outputs", "outputs/predictions", "outputs/checkpoints",
                 "outputs/metrics", "outputs/plots", "outputs/pilots"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROOT


def cfg():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_csv(path, rows, cols=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    cols = cols or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_id(source, lang, split, i):
    return f"{source}:{lang}:{split}:{i}"


def normalize_label(text, labels):
    """Accept only a complete allowed class name."""
    s = unicodedata.normalize("NFKC", str(text or "")).casefold().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\s\"'`.,:;!?\[\](){}]+|[\s\"'`.,:;!?\[\](){}]+$", "", s)
    accepted = {re.sub(r"\s+", " ", label.casefold().strip()): label for label in labels}
    return accepted.get(s)


def stratified_prefix_samples(rows, sizes, seed):
    """Return nested, balanced training subsets for each examples-per-class size."""
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    if not groups:
        raise ValueError("Training examples are empty")
    shuffled = {}
    for label in sorted(groups):
        items = sorted(groups[label], key=lambda x: str(x["example_id"]))
        random.Random(f"{seed}:{label}").shuffle(items)
        shuffled[label] = items
    output = {}
    for n in sizes:
        insufficient = {k: len(v) for k, v in shuffled.items() if len(v) < n}
        if insufficient:
            output[n] = {"available": False, "shortfall": insufficient, "rows": []}
            continue
        subset = [row for label in sorted(shuffled) for row in shuffled[label][:n]]
        random.Random(f"{seed}:{n}:ordering").shuffle(subset)
        output[n] = {"available": True, "rows": subset, "shortfall": {}}
    return output


def validate_split(rows, labels):
    seen_ids, splits_by_text, split_by_lang_index = set(), {}, {}
    problems = []
    for r in rows:
        if not (str(r["text"]).strip() and str(r["label"]) in labels):
            problems.append("Empty text or unsupported label")
        if r["example_id"] in seen_ids:
            problems.append("Duplicate example_id")
        seen_ids.add(r["example_id"])
        key = (r["lang"], unicodedata.normalize("NFKC", str(r["text"]).strip()).casefold())
        splits_by_text.setdefault(key, set()).add(r["split"])
        if r.get("index_id") not in (None, ""):
            key2 = (r["lang"], str(r["index_id"]))
            split_by_lang_index.setdefault(key2, set()).add(r["split"])
    for key, seen in splits_by_text.items():
        if len(seen) > 1:
            problems.append(f"Same-language text across splits: {key[0]}")
    for key, seen in split_by_lang_index.items():
        if len(seen) > 1:
            problems.append(f"Same-language index_id across splits: {key[0]}")
    return sorted(set(problems))


def ensure_task(task):
    folder = ROOT / "data" / task
    manifest = folder / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Run preprocessing first or provide an audited dataset in {folder}")
    return read_json(manifest)


def load_task(task="sib200", lang="ur", split="train"):
    ensure_task(task)
    path = ROOT / "data" / task / f"{lang}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


def task_labels(task="sib200"):
    return ensure_task(task)["labels"]


def dataset_sha(task, lang, split):
    return file_hash(ROOT / "data" / task / f"{lang}_{split}.csv")


def examples_for(task, n, seed, lang="ur"):
    subset = stratified_prefix_samples(load_task(task, lang, "train"), [n], seed)[n]
    if not subset["available"]:
        raise ValueError(f"Not enough examples per class for n={n}: {subset['shortfall']}")
    return subset["rows"]


def instruction(text, labels, lang="ur"):
    permitted = ", ".join(labels)
    if lang == "en":
        return (f"Choose exactly one topic label from: {permitted}. "
                f"Reply with ONLY the label.\nText: {text}")
    return ("دی گئی عبارت کا موضوع منتخب کریں۔ صرف انگریزی لیبل لکھیں، وضاحت نہیں۔\n"
            f"Labels: {permitted}\nمتن: {text}")


def messages_for(text, labels, lang="ur", demonstrations=None):
    messages = [{"role": "system", "content":
                 "You are a topic classifier. Respond with exactly one allowed English category label, and nothing else."}]
    for demo in demonstrations or []:
        messages.extend([
            {"role": "user", "content": instruction(demo["text"], labels, lang)},
            {"role": "assistant", "content": demo["label"]},
        ])
    messages.append({"role": "user", "content": instruction(text, labels, lang)})
    return messages


def numeric_token_ids(value, context="token IDs"):
    """Flatten valid token IDs, including a Hugging Face BatchEncoding."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"{context}: dictionary has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context}: expected a sequence of integers, got {type(value).__name__}")
    invalid = next(((i, repr(x)) for i, x in enumerate(value)
                    if isinstance(x, bool) or not isinstance(x, int)), None)
    if invalid is not None:
        raise ValueError(f"{context}: non-integer token at position {invalid[0]}: {invalid[1]}")
    return list(value)


def encode_supervised(tokenizer, text, label, labels, lang="ur", max_length=256):
    """Train only on the assistant's answer, without truncating its label."""
    def encode_text(candidate):
        prompt = messages_for(candidate, labels, lang)
        prefix = numeric_token_ids(
            tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True),
            context="chat prompt input_ids",
        )
        prefix_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            prompt + [{"role": "assistant", "content": str(label)}],
            tokenize=False, add_generation_prompt=False,
        )
        if not isinstance(prefix_text, str) or not isinstance(full_text, str):
            raise ValueError("Chat tokenizer must support tokenize=False to render text")
        if not full_text.startswith(prefix_text):
            raise ValueError("Rendered chat template is not prefix-aligned")
        suffix_text = full_text[len(prefix_text):]
        if not suffix_text.startswith(str(label)):
            raise ValueError("Assistant answer boundary is incorrect")
        suffix = numeric_token_ids(tokenizer.encode(suffix_text, add_special_tokens=False),
                                   context="assistant response input_ids")
        if not suffix:
            raise ValueError("Chat template generated no response tokens")
        return prefix, prefix + suffix

    source_ids = numeric_token_ids(tokenizer.encode(str(text), add_special_tokens=False),
                                   context="source text input_ids")
    prefix, full = encode_text(str(text))
    kept = len(source_ids)
    if len(full) > max_length:
        low, high, best = 0, len(source_ids), None
        while low <= high:
            middle = (low + high) // 2
            candidate = tokenizer.decode(source_ids[:middle], skip_special_tokens=True)
            p, f = encode_text(candidate)
            if len(f) <= max_length:
                best = (middle, p, f)
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError(f"Prompt+target alone exceeds max_length={max_length}. Increase config.max_length.")
        kept, prefix, full = best
    if not len(prefix) < len(full) <= max_length:
        raise ValueError("Missing answer supervision or excessive sequence length")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prefix) + full[len(prefix):],
        "source_truncated": kept < len(source_ids),
        "removed_source_tokens": len(source_ids) - kept,
    }


def supervised_collator(tokenizer):
    """Pad examples; prompt and padding labels remain -100."""
    def collate(batch):
        import torch
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        width = max(len(item["input_ids"]) for item in batch)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer has no pad or EOS token")
        ids, attention, labels = [], [], []
        for i, item in enumerate(batch):
            i_ids = numeric_token_ids(item["input_ids"], f"row {i} input_ids")
            i_attention = numeric_token_ids(item["attention_mask"], f"row {i} attention_mask")
            i_labels = numeric_token_ids(item["labels"], f"row {i} labels")
            if len(i_ids) != len(i_attention) or len(i_ids) != len(i_labels):
                raise ValueError(f"Row {i}: token, attention and label lengths differ")
            if not any(v != -100 for v in i_labels):
                raise ValueError(f"Row {i}: no supervised answer tokens")
            extra = width - len(i_ids)
            ids.append(i_ids + [pad] * extra)
            attention.append(i_attention + [0] * extra)
            labels.append(i_labels + [-100] * extra)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def model_load_dtype(torch_module, train_mode=False, training_method=None):
    """Full FP16 AMP needs FP32 trainable weights; frozen LoRA base may use FP16."""
    if training_method not in (None, "lora", "full"):
        raise ValueError(f"Unknown training method: {training_method}")
    if train_mode and training_method == "full":
        return torch_module.float32
    return torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32


def load_model(model_id, adapter_path=None, train_mode=False, training_method=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = model_load_dtype(torch, train_mode=train_mode, training_method=training_method)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=train_mode)
    if torch.cuda.is_available():
        model.to("cuda")
    if not train_mode:
        model.config.use_cache = True
        model.eval()
    return model, tokenizer


def run_generation(model, tokenizer, row, labels, lang="ur", demonstrations=None, max_new_tokens=16):
    """Generate a label with the same prompt as training."""
    import torch
    prompt = messages_for(row["text"], labels, lang, demonstrations)
    encoded = tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                            tokenize=True, return_tensors="pt")
    token_ids = numeric_token_ids(encoded, "inference chat input_ids")
    if not token_ids:
        raise ValueError("Chat template returned no inference tokens")
    device = next(model.parameters()).device
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(tokens)
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(
            input_ids=tokens, attention_mask=attention, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=(tokenizer.pad_token_id
                                           if tokenizer.pad_token_id is not None
                                           else tokenizer.eos_token_id),
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(result[0][tokens.shape[1]:], skip_special_tokens=True).strip()
    label = normalize_label(decoded, labels)
    return {
        "raw_prediction": decoded, "prediction": label or "", "valid": bool(label),
        "seconds": round(time.perf_counter() - start, 5),
    }


def evaluate_predictions(rows, predictions, labels):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    expected = {r["example_id"]: r for r in rows}
    ids = [r["example_id"] for r in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction example IDs")
    if set(ids) != set(expected):
        raise ValueError("Prediction IDs mismatched: missing=%d, unknown=%d" % (
            len(set(expected) - set(ids)), len(set(ids) - set(expected))))
    y_true = [expected[i]["label"] for i in ids]
    y_pred = [p["prediction"] if p["prediction"] in labels else "__INVALID__"
              for p in predictions]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        "invalid_count": sum(p not in labels for p in y_pred),
        "count": len(rows),
        "correct_count": sum(a == b for a, b in zip(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                       output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=labels + ["__INVALID__"]).tolist(),
        "confusion_labels": labels + ["__INVALID__"],
    }


def tokenize_dataset(rows, tokenizer, labels, lang, max_length):
    from torch.utils.data import Dataset
    class EncodedRows(Dataset):
        def __init__(self):
            self.items = [encode_supervised(tokenizer, r["text"], r["label"], labels,
                                            lang, max_length) for r in rows]
        def __len__(self):
            return len(self.items)
        def __getitem__(self, index):
            return self.items[index]
    return EncodedRows()


def training_arguments_kwargs(c, method, out, seed, epochs, max_steps, pilot, has_cuda,
                               training_args_class=None, early_stopping=False):
    """Prepare arguments for Kaggle Transformers 5 and optional epoch-level stopping."""
    import inspect
    if training_args_class is None:
        from transformers import TrainingArguments as training_args_class
    supported = set(inspect.signature(training_args_class).parameters)
    if method not in ("lora", "full"):
        raise ValueError(f"Unknown training method: {method}")
    if early_stopping and (pilot or max_steps is not None):
        raise ValueError("Early stopping is for complete-epoch training, not pilots/max_steps")
    options = {
        "output_dir": str(out),
        "num_train_epochs": epochs if epochs is not None else c["train_epochs"],
        "max_steps": max_steps if max_steps is not None else -1,
        "per_device_train_batch_size": c["per_device_batch_size"],
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": c["gradient_accumulation"],
        "learning_rate": c["learning_rate_lora"] if method == "lora" else c["learning_rate_full"],
        "optim": c["optimizer"],
        "max_grad_norm": c["max_grad_norm"],
        "fp16": bool(has_cuda), "bf16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": "no" if pilot else "steps",
        "save_steps": 10,
        "save_total_limit": 1,
        "logging_steps": 1 if pilot else 10,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": seed, "data_seed": seed,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0.03,
        "eval_strategy": "no",
        "train_sampling_strategy": "random",
        "ddp_find_unused_parameters": False,
    }
    if early_stopping:
        options.update({
            "eval_strategy": "epoch", "save_strategy": "epoch", "logging_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss", "greater_is_better": False,
            "save_total_limit": 3,
        })
    missing = sorted(set(options) - supported)
    if missing:
        raise RuntimeError("TrainingArguments unsupported by installed Transformers 5: " + ", ".join(missing))
    return options


def trainer_processing_arg(trainer_class, tokenizer):
    import inspect
    if "processing_class" not in inspect.signature(trainer_class).parameters:
        raise RuntimeError("Trainer needs the Transformers 5 processing_class API")
    return {"processing_class": tokenizer}


def check_training_api():
    """Check Hugging Face API compatibility without downloading a model."""
    import importlib.metadata as md
    import sys
    import torch
    import transformers
    import peft
    from transformers import Trainer, TrainingArguments
    from packaging.version import Version
    if sys.version_info[:2] != (3, 13):
        print("Notice: project tested on Python 3.13, current:", sys.version.split()[0])
    if Version(transformers.__version__).major != 5:
        raise RuntimeError("Project targets Transformers 5.x, found " + transformers.__version__)
    if Version(peft.__version__) < Version("0.20"):
        raise RuntimeError("Project needs PEFT >=0.20, found " + peft.__version__)
    options = training_arguments_kwargs(cfg(), "lora", ROOT / "outputs" / "api_check", 42,
                                        1, 2, True, torch.cuda.is_available(), TrainingArguments)
    args = TrainingArguments(**options)
    processing_key = next(iter(trainer_processing_arg(Trainer, None)))
    return {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.cuda.is_available(), "transformers": transformers.__version__,
        "peft": peft.__version__, "accelerate": md.version("accelerate"),
        "training_arguments": sorted(options), "sampling_key": "train_sampling_strategy",
        "warmup_key": "warmup_steps", "trainer_tokenizer_key": processing_key,
        "optim": str(args.optim),
    }


def train_run(method, task, size, seed, run_name=None, lang="ur", rank=8,
              max_steps=None, pilot=False, model_override=None, epochs=None,
              early_stopping=False, early_stopping_patience=2,
              early_stopping_threshold=0.001):
    """Train LoRA or full tuning; optionally stop by validation loss, save best epoch.

    Do not evaluate on test during training. Always start a new run name when
    changing the early-stopping configuration or training data.
    """
    import math
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model

    c = cfg()
    if early_stopping:
        if pilot or max_steps is not None:
            raise ValueError("Early stopping requires pilot=False and max_steps=None")
        if isinstance(early_stopping_patience, bool) or not isinstance(early_stopping_patience, int) or early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be a positive integer number of epochs")
        if not isinstance(early_stopping_threshold, (int, float)) or not math.isfinite(early_stopping_threshold) or not 0 <= early_stopping_threshold < 1:
            raise ValueError("early_stopping_threshold must be between 0 and 1")
        # 'epochs' is only a safety limit. Early stopping determines the real end.
        if epochs is None:
            epochs = 20
        if epochs < 2:
            raise ValueError("Early stopping needs a safety cap of at least 2 epochs")

    # Catch API errors before downloading the model.
    training_arguments_kwargs(c, method, ROOT / "outputs" / "api_check", seed,
                              epochs, max_steps, pilot, torch.cuda.is_available(),
                              TrainingArguments, early_stopping=early_stopping)
    trainer_processing_arg(Trainer, None)
    set_seed(seed)
    folders()

    model_id = model_override or c.get("english_anchor") or c["model_id"]
    rows = examples_for(task, size, seed, lang=lang)
    val_rows = load_task(task, lang, "validation")
    labels = task_labels(task)
    safe_name = run_name or (f"{task}_{method}_{lang}_n{size}_s{seed}"
                             + (f"_r{rank}" if method == "lora" else "")
                             + ("_es2" if early_stopping else ""))
    out = ROOT / "outputs" / ("pilots" if pilot else "checkpoints") / safe_name
    if (out / "complete.json").exists():
        raise FileExistsError(f"Existing finished run, refusing overwrite: {out}")

    from importlib import metadata as package_metadata
    runtime_versions = {name: package_metadata.version(name)
                        for name in ("transformers", "peft", "torch", "accelerate")}
    spec = {
        "runtime_versions": runtime_versions, "method": method, "task": task,
        "size": size, "seed": seed, "lang": lang, "rank": rank,
        "model_id": model_id, "config": c, "max_steps": max_steps, "epochs": epochs,
        "early_stopping": bool(early_stopping),
        "early_stopping_patience": early_stopping_patience if early_stopping else None,
        "early_stopping_threshold": early_stopping_threshold if early_stopping else None,
        "train_sha": dataset_sha(task, lang, "train"),
        "val_sha": dataset_sha(task, lang, "validation"),
    }
    previous = out / "run_spec.json"
    existing_checkpoints = sorted(
        out.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    resume_checkpoint = None
    if out.exists() and any(out.iterdir()):
        if not previous.exists() or read_json(previous) != spec:
            raise FileExistsError(f"Partial run metadata differs; refusing unsafe resume: {out}")
        if not existing_checkpoints:
            leftover = {p.name for p in out.iterdir()} - {"run_spec.json", "session_history.jsonl"}
            if leftover:
                raise FileExistsError(f"Unexpected partial files in {out}: {leftover}")
            print("No checkpoint saved before interruption; training restarts from step zero.")
        else:
            resume_checkpoint = existing_checkpoints[-1]
    out.mkdir(parents=True, exist_ok=True)
    if not previous.exists():
        write_json(previous, spec)

    model, tokenizer = load_model(model_id, train_mode=True, training_method=method)
    if method == "full":
        wrong = sorted({str(p.dtype) for p in model.parameters()
                        if p.requires_grad and p.dtype != torch.float32})
        if wrong:
            raise RuntimeError("Full FP16-AMP requires FP32 trainable model weights, found " + str(wrong))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if method == "lora":
        adapter = LoraConfig(
            r=rank, lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
            target_modules=["q_proj", "v_proj"], bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, adapter)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    elif method != "full":
        raise ValueError(method)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    train_dataset = tokenize_dataset(rows, tokenizer, labels, lang, c["max_length"])
    validation = (tokenize_dataset(val_rows, tokenizer, labels, lang, c["max_length"])
                  if early_stopping else None)
    if not len(train_dataset):
        raise ValueError("No training examples")
    if early_stopping and (validation is None or not len(validation)):
        raise ValueError("Early stopping requires a non-empty validation split")
    truncated = sum(item["source_truncated"] for item in train_dataset.items)
    removed_tokens = sum(item["removed_source_tokens"] for item in train_dataset.items)
    options = training_arguments_kwargs(
        c, method, out, seed, epochs, max_steps, pilot, torch.cuda.is_available(),
        TrainingArguments, early_stopping=early_stopping,
    )
    args = TrainingArguments(**options)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=validation,
        data_collator=supervised_collator(tokenizer),
        **trainer_processing_arg(Trainer, tokenizer),
    )
    if early_stopping:
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        ))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    history_path = out / "session_history.jsonl"
    try:
        output = trainer.train(resume_from_checkpoint=str(resume_checkpoint)
                               if resume_checkpoint else None)
    except BaseException as exc:
        append_jsonl(history_path, {
            "status": "interrupted", "wall_seconds": round(time.perf_counter() - start, 2),
            "error": type(exc).__name__,
            "resume_from": str(resume_checkpoint) if resume_checkpoint else None,
        })
        raise
    wall_this_session = time.perf_counter() - start
    memory = (round(torch.cuda.max_memory_allocated() / 1024**3, 3)
              if torch.cuda.is_available() else None)
    append_jsonl(history_path, {
        "status": "complete_session", "wall_seconds": round(wall_this_session, 2),
        "peak_gpu_allocated_gib": memory, "resumed": resume_checkpoint is not None,
    })
    sessions = read_jsonl(history_path)
    wall = sum(entry.get("wall_seconds", 0) for entry in sessions)
    recorded_memory = [entry["peak_gpu_allocated_gib"] for entry in sessions
                       if entry.get("peak_gpu_allocated_gib") is not None]
    if recorded_memory:
        memory = max(recorded_memory)

    validation_history = [
        {"epoch": float(entry.get("epoch", 0)), "step": int(entry.get("step", 0)),
         "eval_loss": float(entry["eval_loss"])}
        for entry in trainer.state.log_history if "eval_loss" in entry
    ]
    finite = [entry for entry in validation_history if math.isfinite(entry["eval_loss"])]
    best = min(finite, key=lambda entry: entry["eval_loss"]) if finite else None
    best_checkpoint = trainer.state.best_model_checkpoint
    epochs_completed = float(trainer.state.epoch or 0)
    if early_stopping:
        if not best or not best_checkpoint:
            raise RuntimeError("No saved best validation checkpoint. Do not use this run for comparison.")
        write_json(out / "validation_loss_history.json", validation_history)

    # Trainer has automatically reloaded the lowest-validation-loss checkpoint.
    trainer.save_model(str(out / "model"))
    tokenizer.save_pretrained(str(out / "model"))
    summary = {
        "run_id": safe_name, "status": "complete", "task": task, "method": method,
        "language": lang, "examples": len(rows), "per_class": size,
        "seed": seed, "lora_rank": rank if method == "lora" else None,
        "base_model": model_id, "trainable_params": trainable,
        "total_params": total, "trainable_pct": round(100 * trainable / total, 5),
        "optimizer": c["optimizer"], "max_length": c["max_length"],
        "wall_seconds": round(wall, 2), "peak_gpu_allocated_gib": memory,
        "global_steps": output.global_step, "train_loss": output.training_loss,
        "resume_count": sum(bool(entry.get("resumed")) for entry in sessions),
        "wall_seconds_this_session": round(wall_this_session, 2),
        "train_csv_sha256": dataset_sha(task, lang, "train"),
        "val_csv_sha256": dataset_sha(task, lang, "validation"),
        "truncated_training_examples": truncated,
        "source_tokens_removed": removed_tokens,
        "saved_path": str(out / "model"), "pilot": pilot,
        "config": c, "runtime_versions": runtime_versions,
        "early_stopping": bool(early_stopping),
        "early_stopping_patience": early_stopping_patience if early_stopping else None,
        "early_stopping_threshold": early_stopping_threshold if early_stopping else None,
        "epochs_completed": round(epochs_completed, 4),
        "best_epoch": round(best["epoch"], 4) if best else None,
        "best_validation_loss": best["eval_loss"] if best else None,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "validation_loss_history_file": (str(out / "validation_loss_history.json")
                                         if early_stopping else None),
    }
    write_json(out / "complete.json", summary)
    append_jsonl(ROOT / "outputs/metrics/train_runs.jsonl", summary)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def infer_rows(model, tokenizer, rows, labels, lang, max_new_tokens=16, demonstrations=None):
    return [dict(example_id=row["example_id"], reference=row["label"],
                 **run_generation(model, tokenizer, row, labels, lang, demonstrations, max_new_tokens))
            for row in rows]


def save_prediction_run(path, predictions, metadata):
    path = Path(path)
    write_csv(path, predictions,
              cols=["example_id", "reference", "raw_prediction", "prediction", "valid", "seconds"])
    write_json(path.with_suffix(".meta.json"), metadata)


def validate_prediction_metadata(meta, task, lang, split, model_id=None):
    if meta["dataset_sha256"] != dataset_sha(task, lang, split):
        raise ValueError("Dataset file changed since inference")
    if (meta["task"], meta["lang"], meta["split"]) != (task, lang, split):
        raise ValueError("Wrong task, language, or split in prediction metadata")
    if model_id and meta.get("model_id") != model_id:
        raise ValueError("Wrong model reference")
    return True


def infer_resume(model, tokenizer, rows, labels, lang, path, metadata, max_new_tokens=16):
    """Persist predictions as they are generated and safely resume after interruption."""
    path = Path(path)
    progress = path.with_suffix(".progress.jsonl")
    progress_meta = path.with_suffix(".progress.meta.json")
    final_meta = path.with_suffix(".meta.json")
    fields = ["task", "lang", "split", "dataset_sha256", "model_id", "run_id",
              "method", "subset", "decoding"]
    def compatible(old):
        if any(old.get(key) != metadata.get(key) for key in fields):
            raise ValueError("Saved inference uses a different model/dataset/split/config")
    expected = {row["example_id"] for row in rows}
    if path.exists() and final_meta.exists():
        compatible(read_json(final_meta))
        completed = read_csv(path)
        if len(completed) == len(expected) and {row["example_id"] for row in completed} == expected:
            print("Completed inference already exists:", path)
            return completed
    if progress.exists():
        if not progress_meta.exists():
            raise ValueError("Progress file is missing its metadata")
        compatible(read_json(progress_meta))
    else:
        write_json(progress_meta, metadata)
    previous = read_jsonl(progress)
    done = {row["example_id"]: row for row in previous}
    if len(done) != len(previous):
        raise ValueError("Duplicate IDs in saved inference")
    if not set(done).issubset(expected):
        raise ValueError("Partial inference has unknown IDs")
    for i, row in enumerate(rows):
        if row["example_id"] in done:
            continue
        prediction = dict(example_id=row["example_id"], reference=row["label"],
                          **run_generation(model, tokenizer, row, labels, lang,
                                           max_new_tokens=max_new_tokens))
        append_jsonl(progress, prediction)
        done[prediction["example_id"]] = prediction
        if (i + 1) % 20 == 0:
            print("Inference:", len(done), "/", len(rows))
    if set(done) != expected:
        raise ValueError("Incomplete inference; refusing to publish final CSV")
    ordered = [done[row["example_id"]] for row in rows]
    save_prediction_run(path, ordered, metadata)
    print("Finalized inference:", path)
    return ordered
