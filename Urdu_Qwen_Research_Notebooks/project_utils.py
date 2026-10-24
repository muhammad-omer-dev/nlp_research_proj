"""Shared, deliberately small utilities for the Urdu LoRA versus full-tuning project.

The module avoids importing transformers/torch/datasets at module import time. Data,
metrics and safety tests can run locally without downloading a model or needing a GPU.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
PAPER_LABELS = ["business", "entertainment", "health", "politics", "science", "sports", "other", "world"]


def folders():
    for name in ["data", "outputs", "outputs/predictions", "outputs/checkpoints", "outputs/metrics", "outputs/plots", "outputs/pilots"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROOT


def cfg():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path):
    p = Path(path)
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_csv(path, rows, cols=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    cols = cols or (list(rows[0]) if rows else [])
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError: pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass


def stable_id(source, lang, split, i):
    return f"{source}:{lang}:{split}:{i}"


def normalize_label(text, labels):
    """Strict closed-set parser: no substring guessing or accidental partial matches."""
    s = unicodedata.normalize("NFKC", str(text or "")).casefold().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\s\"'`.,:;!?\[\](){}]+|[\s\"'`.,:;!?\[\](){}]+$", "", s)
    accepted = {re.sub(r"\s+", " ", v.casefold().strip()): v for v in labels}
    # Do NOT map 7 SIB categories to the paper's 8 classes.
    return accepted.get(s, None)


def stratified_prefix_samples(rows, sizes, seed):
    """Nested few-shot subsets. No replacement, exact count per class, train only."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    if not by_label: raise ValueError("Training examples are empty")
    shuffled = {}
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda x: str(x["example_id"]))
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(group)
        shuffled[label] = group
    answer = {}
    for n in sizes:
        insufficient = {k: len(v) for k, v in shuffled.items() if len(v) < n}
        if insufficient:
            answer[n] = {"available": False, "shortfall": insufficient, "rows": []}
            continue
        subset = [v for k in sorted(shuffled) for v in shuffled[k][:n]]
        random.Random(f"{seed}:{n}:ordering").shuffle(subset)
        answer[n] = {"available": True, "rows": subset, "shortfall": {}}
    return answer


def validate_split(rows, labels):
    seen_ids, splits_by_text, split_by_lang_index = set(), {}, {}
    problems = []
    for r in rows:
        if not (str(r["text"]).strip() and str(r["label"]) in labels):
            problems.append("Empty text or unsupported label")
        if r["example_id"] in seen_ids: problems.append("Duplicate example_id")
        seen_ids.add(r["example_id"])
        key = (r["lang"], unicodedata.normalize("NFKC", str(r["text"]).strip()).casefold())
        splits_by_text.setdefault(key, set()).add(r["split"])
        if r.get("index_id") not in (None, ""):
            key2 = (r["lang"], str(r["index_id"]))
            split_by_lang_index.setdefault(key2, set()).add(r["split"])
    for key, v in splits_by_text.items():
        if len(v) > 1: problems.append(f"Same-language text across splits: {key[0]}")
    for key, v in split_by_lang_index.items():
        if len(v) > 1: problems.append(f"Same-language index_id across splits: {key[0]}")
    return sorted(set(problems))


def ensure_task(task):
    d = ROOT / "data" / task
    m = d / "manifest.json"
    if not m.exists(): raise FileNotFoundError(f"Run 01_preprocess_data.ipynb first, or import a reviewed benchmark into {d}.")
    return read_json(m)


def load_task(task="sib200", lang="ur", split="train"):
    ensure_task(task)
    path = ROOT / "data" / task / f"{lang}_{split}.csv"
    if not path.exists(): raise FileNotFoundError(path)
    return read_csv(path)


def task_labels(task="sib200"):
    return ensure_task(task)["labels"]


def dataset_sha(task, lang, split):
    return file_hash(ROOT / "data" / task / f"{lang}_{split}.csv")


def examples_for(task, n, seed, lang="ur"):
    data = load_task(task, lang, "train")
    subset = stratified_prefix_samples(data, [n], seed)[n]
    if not subset["available"]:
        raise ValueError(f"Not enough examples per class for n={n}: {subset['shortfall']}")
    return subset["rows"]


def instruction(text, labels, lang="ur"):
    permitted = ", ".join(labels)
    if lang == "en":
        return f"Choose exactly one topic label from: {permitted}. Reply with ONLY the label.\nText: {text}"
    return f"دی گئی عبارت کا موضوع منتخب کریں۔ صرف انگریزی لیبل لکھیں، وضاحت نہیں۔\nLabels: {permitted}\nمتن: {text}"


def messages_for(text, labels, lang="ur", demonstrations=None):
    messages = [{"role": "system", "content": "You are a topic classifier. Respond with exactly one allowed English category label, and nothing else."}]
    for demo in demonstrations or []:
        messages.extend([
            {"role": "user", "content": instruction(demo["text"], labels, lang)},
            {"role": "assistant", "content": demo["label"]},
        ])
    messages.append({"role": "user", "content": instruction(text, labels, lang)})
    return messages


def encode_supervised(tokenizer, text, label, labels, lang="ur", max_length=256):
    """Chat-template aligned, assistant-only loss. Shortens INPUT TEXT, never target.

    Records how many source tokens are removed; dataset statistics are saved in
    each run metadata. No suffix of the answer is ever silently truncated.
    """
    def encode_text(candidate):
        prompt = messages_for(candidate, labels, lang)
        # Always use EXACTLY the same prompt tokens that inference will use.
        # Tokenizing a completed conversation separately can merge BPE tokens
        # across the prompt/answer boundary, so its token IDs do not have to
        # start with the independently tokenized generation prompt.
        prefix_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        if isinstance(prefix_ids, dict):
            prefix_ids = prefix_ids["input_ids"]
        if hasattr(prefix_ids, "tolist"):
            prefix_ids = prefix_ids.tolist()
        prefix_ids = list(prefix_ids)

        prefix_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            prompt + [{"role": "assistant", "content": str(label)}],
            tokenize=False, add_generation_prompt=False
        )
        if not isinstance(prefix_text, str) or not isinstance(full_text, str):
            raise ValueError("Chat tokenizer must support tokenize=False to render text")
        if not full_text.startswith(prefix_text):
            raise ValueError(
                "Rendered chat template is not prefix-aligned: inspect tokenizer.chat_template. "
                "Do not train until the assistant boundary is verified."
            )
        suffix_text = full_text[len(prefix_text):]
        if not suffix_text.startswith(str(label)):
            raise ValueError(
                "Assistant response is not immediately after the generation prompt; "
                "inspect the model's chat template before training."
            )
        # No extra BOS/EOS: chat templates already provide end-of-turn tokens.
        # Concatenating independently tokenized pieces preserves the generation
        # prompt prefix and allows a precise assistant-only loss mask.
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        if hasattr(suffix_ids, "tolist"):
            suffix_ids = suffix_ids.tolist()
        suffix_ids = list(suffix_ids)
        if not suffix_ids:
            raise ValueError("Chat template generated no response tokens")
        return prefix_ids, prefix_ids + suffix_ids

    original_tokens = tokenizer.encode(str(text), add_special_tokens=False)
    prefix, full = encode_text(str(text))
    kept = len(original_tokens)
    if len(full) > max_length:
        low, high = 0, len(original_tokens)
        best = None
        while low <= high:
            middle = (low + high) // 2
            candidate = tokenizer.decode(original_tokens[:middle], skip_special_tokens=True)
            p, f = encode_text(candidate)
            if len(f) <= max_length:
                best = (middle, p, f)
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError(f"Prompt+target alone exceeds max_length={max_length}. Increase config.max_length.")
        kept, prefix, full = best
    assert len(full) <= max_length and len(full) > len(prefix), "Missing supervised response tokens"
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):],
            "source_truncated": kept < len(original_tokens),
            "removed_source_tokens": len(original_tokens)-kept}


def supervised_collator(tokenizer):
    """Dynamic right padding; output is ready for CausalLM loss."""
    def collate(batch):
        import torch
        width = max(len(x["input_ids"]) for x in batch)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids, attn, labs = [], [], []
        for b in batch:
            extra = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * extra)
            attn.append(b["attention_mask"] + [0] * extra)
            labs.append(b["labels"] + [-100] * extra)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn), "labels": torch.tensor(labs)}
    return collate


def load_model(model_id, adapter_path=None, train_mode=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    kw = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=train_mode)
    if torch.cuda.is_available(): model.to("cuda")
    if not train_mode:
        model.config.use_cache = True
        model.eval()
    return model, tokenizer


def run_generation(model, tokenizer, row, labels, lang="ur", demonstrations=None, max_new_tokens=16):
    import torch
    prompt = messages_for(row["text"], labels, lang, demonstrations)
    encoded = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    if isinstance(encoded, dict): encoded = encoded["input_ids"]
    model_device = next(model.parameters()).device
    encoded = encoded.to(model_device)
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(input_ids=encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(result[0][encoded.shape[1]:], skip_special_tokens=True).strip()
    seconds = time.perf_counter() - start
    label = normalize_label(decoded, labels)
    return {"raw_prediction": decoded, "prediction": label or "", "valid": bool(label), "seconds": round(seconds, 5)}


def evaluate_predictions(rows, predictions, labels):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    expected = {r["example_id"]: r for r in rows}
    ids = [p["example_id"] for p in predictions]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate prediction example IDs")
    if set(ids) != set(expected):
        raise ValueError(f"Prediction IDs mismatched: missing={len(set(expected)-set(ids))}, unknown={len(set(ids)-set(expected))}")
    y_true = [expected[i]["label"] for i in ids]
    y_pred = [p["prediction"] if p["prediction"] in labels else "__INVALID__" for p in predictions]
    correct = [a == b for a, b in zip(y_true, y_pred)]
    # Restrict F1 reported classes to true categories; __INVALID__ still causes FNs.
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "invalid_count": int(sum(p not in labels for p in y_pred)),
            "count": len(rows),
            "correct_count": sum(correct),
            "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels + ["__INVALID__"]).tolist(),
            "confusion_labels": labels + ["__INVALID__"]}


def tokenize_dataset(rows, tokenizer, labels, lang, max_length):
    from torch.utils.data import Dataset
    class EncodedRows(Dataset):
        def __init__(self): self.items = [encode_supervised(tokenizer, r["text"], r["label"], labels, lang, max_length) for r in rows]
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]
    return EncodedRows()


def train_run(method, task, size, seed, run_name=None, lang="ur", rank=8,
              max_steps=None, pilot=False, model_override=None, epochs=None):
    """Train one reproducible model. No test-set access here. LoRA & full share loss/data/optimizer."""
    import torch
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, TaskType, get_peft_model
    c = cfg()
    set_seed(seed)
    folders()
    model_id = model_override or c.get("english_anchor") or c["model_id"]
    rows = examples_for(task, size, seed, lang=lang)
    _ = load_task(task, lang, "validation")  # existence check only; no training on validation
    labels = task_labels(task)
    safe_name = run_name or f"{task}_{method}_{lang}_n{size}_s{seed}" + (f"_r{rank}" if method == "lora" else "")
    out = ROOT / "outputs" / ("pilots" if pilot else "checkpoints") / safe_name
    if (out / "complete.json").exists():
        raise FileExistsError(f"Existing finished run, refusing to overwrite: {out}")
    spec = {"method":method,"task":task,"size":size,"seed":seed,"lang":lang,"rank":rank,
            "model_id":model_id,"config":c,"max_steps":max_steps,"epochs":epochs,
            "train_sha":dataset_sha(task,lang,"train"),
            "val_sha":dataset_sha(task,lang,"validation")}
    previous = out / "run_spec.json"
    existing_checkpoints = sorted(out.glob("checkpoint-*"),
                                 key=lambda p: int(p.name.rsplit("-",1)[-1]) if p.name.rsplit("-",1)[-1].isdigit() else -1)
    resume_checkpoint = None
    if out.exists() and any(out.iterdir()):
        if not previous.exists() or read_json(previous) != spec:
            raise FileExistsError(f"Partial run metadata differs; refusing unsafe resume: {out}")
        if not existing_checkpoints:
            leftover = {x.name for x in out.iterdir()} - {"run_spec.json", "session_history.jsonl"}
            if leftover:
                raise FileExistsError(f"Unexpected partial files; inspect before restarting: {out}, files={leftover}")
            print('No checkpoint was saved before interruption; restarting training from step zero.')
        else:
            resume_checkpoint = existing_checkpoints[-1]
    out.mkdir(parents=True,exist_ok=True)
    if not previous.exists(): write_json(previous,spec)
    model, tokenizer = load_model(model_id, train_mode=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if method == "lora":
        lora = LoraConfig(r=rank, lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
                          target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  # required by some checkpointing/PEFT versions
    elif method != "full": raise ValueError(method)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    dataset = tokenize_dataset(rows, tokenizer, labels, lang, c["max_length"])
    truncated_train = sum(x["source_truncated"] for x in dataset.items)
    removed_train_tokens = sum(x["removed_source_tokens"] for x in dataset.items)
    # Validation labels are not required for training: keep evaluation separate from the trainer.
    validation = None
    if len(dataset) == 0: raise ValueError("No training examples")
    args = TrainingArguments(
        output_dir=str(out), overwrite_output_dir=True, num_train_epochs=epochs or c["train_epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=c["per_device_batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=c["gradient_accumulation"],
        learning_rate=c["learning_rate_lora"] if method == "lora" else c["learning_rate_full"],
        optim=c["optimizer"], max_grad_norm=c["max_grad_norm"],
        fp16=torch.cuda.is_available(), bf16=False, gradient_checkpointing=True,
        eval_strategy="no", save_strategy="no" if pilot else "steps", save_steps=10,
        save_total_limit=1, logging_steps=1 if pilot else 10,
        dataloader_num_workers=0, group_by_length=False, remove_unused_columns=False,
        report_to="none", seed=seed, data_seed=seed, lr_scheduler_type="linear",
        warmup_ratio=0.03, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, eval_dataset=validation,
                      data_collator=supervised_collator(tokenizer), processing_class=tokenizer)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    history_path = out / "session_history.jsonl"
    try:
        output = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    except BaseException as exc:
        spent = time.perf_counter()-t0
        append_jsonl(history_path, {"status":"interrupted", "wall_seconds":round(spent,2),
                                    "error":type(exc).__name__, "resume_from":str(resume_checkpoint) if resume_checkpoint else None})
        raise
    wall_this_session = time.perf_counter() - t0
    mem = round(torch.cuda.max_memory_allocated()/1024**3, 3) if torch.cuda.is_available() else None
    append_jsonl(history_path, {"status":"complete_session","wall_seconds":round(wall_this_session,2),
                                "peak_gpu_allocated_gib":mem,
                                "resumed":resume_checkpoint is not None})
    session_hist=read_jsonl(history_path)
    wall = sum(x.get("wall_seconds",0) for x in session_hist)
    all_mem = [x.get("peak_gpu_allocated_gib") for x in session_hist if x.get("peak_gpu_allocated_gib") is not None]
    if all_mem: mem=max(all_mem)
    trainer.save_model(str(out / "model"))
    tokenizer.save_pretrained(str(out / "model"))
    summary = {"run_id": safe_name, "status":"complete", "task":task, "method":method, "language":lang,
               "examples":len(rows), "per_class":size, "seed":seed, "lora_rank":rank if method=="lora" else None,
               "base_model":model_id, "trainable_params":trainable, "total_params":total,
               "trainable_pct":round(100 * trainable/total, 5), "optimizer":c["optimizer"],
               "max_length":c["max_length"], "wall_seconds":round(wall,2), "peak_gpu_allocated_gib":mem,
               "global_steps":output.global_step, "train_loss":output.training_loss,
               "resume_count":sum(bool(x.get("resumed")) for x in session_hist),
               "wall_seconds_this_session":round(wall_this_session,2),
               "train_csv_sha256":dataset_sha(task, lang, "train"), "val_csv_sha256":dataset_sha(task, lang, "validation"),
               "truncated_training_examples": truncated_train, "source_tokens_removed": removed_train_tokens,
               "saved_path":str(out / "model"), "pilot":pilot, "config":c}
    write_json(out / "complete.json", summary)
    append_jsonl(ROOT / "outputs/metrics/train_runs.jsonl", summary)
    del trainer, model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summary


def infer_rows(model, tokenizer, rows, labels, lang, max_new_tokens=16, demonstrations=None):
    return [dict(example_id=r["example_id"], reference=r["label"], **run_generation(model, tokenizer, r, labels, lang, demonstrations, max_new_tokens)) for r in rows]


def save_prediction_run(path, predictions, metadata):
    path = Path(path)
    write_csv(path, predictions, cols=["example_id","reference","raw_prediction","prediction","valid","seconds"])
    write_json(path.with_suffix(".meta.json"), metadata)


def validate_prediction_metadata(meta, task, lang, split, model_id=None):
    expected_sha = dataset_sha(task, lang, split)
    if meta["dataset_sha256"] != expected_sha:
        raise ValueError("Dataset file changed since inference; predictions cannot be evaluated against it.")
    if (meta["task"], meta["lang"], meta["split"]) != (task, lang, split):
        raise ValueError("Wrong task, language or split in prediction metadata")
    if model_id and meta.get("model_id") != model_id: raise ValueError("Wrong model reference")
    return True


def infer_resume(model, tokenizer, rows, labels, lang, path, metadata, max_new_tokens=16):
    """Resume inference safely after Colab/Kaggle interruption.

    Every result is persisted to .progress.jsonl; final CSV is created only if all
    requested source IDs are covered. A run with changed dataset/model/task is rejected.
    """
    path = Path(path)
    progress = path.with_suffix('.progress.jsonl')
    progress_meta = path.with_suffix('.progress.meta.json')
    final_meta = path.with_suffix('.meta.json')
    fields = ['task','lang','split','dataset_sha256','model_id','run_id','method','subset','decoding']
    def compatible(existing):
        if any(existing.get(k) != metadata.get(k) for k in fields):
            raise ValueError('Existing inference has a different model/dataset/split/config; choose a new output name.')
    expected = {r['example_id'] for r in rows}
    if path.exists() and final_meta.exists():
        compatible(read_json(final_meta))
        done = read_csv(path)
        if len(done) == len(expected) and {x['example_id'] for x in done} == expected:
            print('Completed inference already exists. Skipping generation:',path)
            return done
    if progress.exists():
        if not progress_meta.exists(): raise ValueError('Progress data exists without metadata')
        compatible(read_json(progress_meta))
    else:
        write_json(progress_meta, metadata)
    existing = read_jsonl(progress)
    done = {r['example_id']: r for r in existing}
    if len(done) != len(existing): raise ValueError('Duplicate IDs in partial inference results')
    if not set(done).issubset(expected):
        raise ValueError('Saved progress contains IDs outside requested evaluation split')
    for j,r in enumerate(rows):
        if r['example_id'] in done: continue
        pred = dict(example_id=r['example_id'], reference=r['label'],
                    **run_generation(model,tokenizer,r,labels,lang,max_new_tokens=max_new_tokens))
        append_jsonl(progress,pred)
        done[pred['example_id']] = pred
        if (j+1)%20==0: print('Inference:',len(done),'/',len(rows),'examples persisted')
    if set(done) != expected: raise ValueError('Incomplete inference; refusing to publish final CSV')
    ordered=[done[r['example_id']] for r in rows]
    save_prediction_run(path,ordered,metadata)
    print('Finalized complete inference; safe to remove progress files after backup.')
    return ordered
