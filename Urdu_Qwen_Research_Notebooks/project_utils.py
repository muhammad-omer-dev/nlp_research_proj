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
