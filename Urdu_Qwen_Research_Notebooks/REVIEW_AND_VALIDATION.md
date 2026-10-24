# Review and validation — 24 October 2026

## Verified locally

- Every shipped notebook parses as Python code and satisfies `nbformat.validate`.
- The first notebook runs on a CPU-only local environment with GPU-only tests disabled.
- Nine unit tests cover strict label parsing, stable nested low-resource sample selection, duplicated source IDs, cross-split text leakage, loss-mask alignment, unacceptable prompt-only input, incomplete prediction rejection, train-only few-shot construction and save/read integrity.
- No simulated F1 values or imaginary training logs are inserted.

## Issues found and repaired during review

1. The original paper's eight-category label set is incompatible with SIB-200's seven-category set. The additional paper benchmark now has a separate task ID, dataset importer and required safety review.
2. Zero/few-shot filenames were initially inconsistent with the evaluator's file selection. Now they end with `_ur_validation.csv` / `_ur_test.csv` and are picked up by notebook 06.
3. Urdu source texts can exceed the maximum token budget. Training now shortens source text only, always preserves the target label, and logs number/extent of truncated texts.
4. Invalid text generations can look like plausible categories when substring matching is used. The parser now matches exact category labels only, after minimal punctuation/whitespace normalization.
5. Incomplete inference output could be mistaken for a complete evaluation. The evaluator now rejects partial sets and requires exact ID coverage and matching SHA256 metadata.
6. Interrupted inference previously lost all outputs. Inference now writes individual progress records and resumes safely.
7. Interrupted training needed a recovery path. Full/LoRA normal runs now save periodic Trainer checkpoints, record run specs and resume only with matching training configuration.
8. LoRA input embeddings and gradient checkpointing can conflict in some PEFT versions. The LoRA setup now calls `enable_input_require_grads` when available.
9. An optimizer mismatch could bias resource comparisons. Both LoRA and full use Adafactor; learning rates remain method-specific and must be tuned fairly.
10. The English 'forgetting' claim was previously underspecified. An optional shared English-trained starting checkpoint is provided to support a more meaningful comparison.
11. The published topic-test file contains only held-out evaluation examples, not a legal training set. A separate notebook uses the cited headline source for train/validation and excludes exact overlaps with test headlines.

## Remaining checks requiring actual Kaggle/Colab GPU

- Install pinned Hugging Face library versions on the hosted CUDA-PyTorch environment.
- Download and load Qwen2.5-0.5B-Instruct and both SIB-200 language configurations.
- Verify the real Qwen chat tokenizer aligns prompt and assistant labels.
- Run LoRA and full 2-step pilots with real CUDA memory measurements; check precision/optimizer compatibility.
- Train both configurations, evaluate real predictions and verify that all checkpoints reload correctly.
- Inspect original XLSX headline columns and confirm `Weird` → `other` mapping before paper task training.
- Validate overlap removal and possible near duplicates in the original headlines source.

**Conclusion:** This is an implementation-complete experimental starter and validated offline notebook suite, **not** evidence that model training has already succeeded on Kaggle or Colab. GPU measurements and scientific results must be produced by executing the experiments.
