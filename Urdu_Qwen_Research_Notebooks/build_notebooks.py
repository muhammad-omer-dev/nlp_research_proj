"""Rebuild numbered Jupyter notebooks from reviewed source cells."""
from pathlib import Path
import nbformat as nbf
R = Path(__file__).resolve().parent
N = R / 'notebooks'; N.mkdir(exist_ok=True)


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())
def make(filename, name, goal, cells):
    preface = [md(f'# {name}\n\n**Purpose:** {goal}\n\n**Platform:** Kaggle or Colab (Python + GPU for model stages). Never use held-out test data for model selection.'),
    code('''# GitHub setup: run this as the FIRST code cell in Kaggle or Colab.
# This clones the source code only; GPU checkpoints/results need separate storage.
from pathlib import Path
import os, sys, subprocess

REPO_URL = "https://github.com/muhammad-omer-dev/nlp_research_proj.git"
REPO_FOLDER = "nlp_research_proj"
PROJECT_FOLDER = "Urdu_Qwen_Research_Notebooks"
PULL_LATEST = False  # True: update an existing clean clone from GitHub

if Path("/kaggle/working").is_dir():
    workspace = Path("/kaggle/working")
elif Path("/content").is_dir():
    workspace = Path("/content")
else:
    workspace = Path.cwd()

repo_dir = workspace / REPO_FOLDER
project_dir = repo_dir / PROJECT_FOLDER

# Local development support: use the existing source folder outside Kaggle/Colab.
local_project = next((p for p in [Path.cwd(), *Path.cwd().parents]
                      if (p / "project_utils.py").is_file()), None)
if local_project is not None and not (Path("/kaggle/working").is_dir() or Path("/content").is_dir()):
    ROOT = local_project
else:
    if not repo_dir.exists():
        print("Cloning:", REPO_URL)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    elif PULL_LATEST:
        # Only fast-forward; protects existing local changes from being overwritten.
        subprocess.check_call(["git", "-C", str(repo_dir), "pull", "--ff-only"])
    else:
        print("Using existing clone:", repo_dir)

    ROOT = project_dir

if not (ROOT / "project_utils.py").is_file():
    raise FileNotFoundError(
        f"Cannot find project_utils.py at {ROOT}. Confirm the GitHub folder name."
    )

os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_utils import *
folders()
print("Project loaded from:", ROOT)
''')]
    nb = nbf.v4.new_notebook(cells=preface+cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info':{'name':'python'}})
    nbf.validate(nb)
    nbf.write(nb, N/filename)

make('00_environment_and_pilot.ipynb', '00 — Environment tests and GPU pilot',
     'Check free GPU access, local safety tests, packages, tokenizer alignment, and optional 2-step LoRA/full timing pilots.', [
md('''## A. Environment and safety checks

Run this notebook **first**. The first three cells work without downloading Qwen. If packages are missing, run the installation cell in Kaggle/Colab (restart the kernel once after installation). This notebook does not generate paper results.'''),
code('''import sys, platform, importlib.util
print('Python:', sys.version.split()[0], 'OS:', platform.platform())
for pkg in ['torch','transformers','peft','accelerate','datasets','sklearn','numpy','pandas']:
    spec = importlib.util.find_spec(pkg)
    print(f'{pkg:15}', 'installed' if spec else 'MISSING')
try:
    import torch
    print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GB:',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))
except ImportError:
    print('Install PyTorch in Kaggle/Colab using the platform-default build.')'''),
md('''### Install only if packages are missing

Kaggle: turn **Internet ON** and **GPU ON**. Colab: Runtime → Change runtime type → T4 GPU. Pip should not replace the platform's CUDA-aware PyTorch.'''),
code('''# Run this cell ONCE in Kaggle/Colab if libraries are missing. Then restart the kernel.
INSTALL_PACKAGES = False
if INSTALL_PACKAGES:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'requirements.txt')])
    print('Restart kernel once before running the model cells.')'''),
code('''# Offline unit tests: no model download or GPU required.
import subprocess
command = [sys.executable, '-m', 'pytest', '-q', str(ROOT/'tests')]
res = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
print(res.stdout[-4000:], res.stderr[-1000:]); assert res.returncode == 0, 'Unit tests failed' '''),
md('''## B. Optional tokenizer and model smoke test

Set `RUN_MODEL_SMOKE=True` only once packages are installed and internet is available. This loads the real 0.5B model and checks that exactly the answer tokens receive loss.'''),
code('''RUN_MODEL_SMOKE = False
if RUN_MODEL_SMOKE:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg()['model_id'])
    labels = DEFAULT_LABELS
    example = encode_supervised(tokenizer, 'پاکستان نے کرکٹ میچ جیت لیا', 'sports', labels, 'ur', max_length=256)
    assert len(example['input_ids']) == len(example['labels'])
    assert any(x != -100 for x in example['labels'])
    print('Prompt tokens ignored:', sum(x == -100 for x in example['labels']))
    print('Supervised tokens:',sum(x != -100 for x in example['labels']))
    model, tok = load_model(cfg()['model_id'])
    import torch
    with torch.no_grad():
        batch = supervised_collator(tok)([example]); batch = {k:v.to(next(model.parameters()).device) for k,v in batch.items()}
        loss = model(**batch).loss
    print('Real model loss:',float(loss)); assert torch.isfinite(loss).item()
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None'''),
md('''## C. Optional two-step GPU training pilot

Run `01_preprocess_data` first. **This is not a scientific result**: it measures real training speed and checks whether full fine-tuning fits. Test LoRA first, then full. Stop and reduce `config.json → max_length` to 192 or 128 if full training runs out of GPU memory. Use two training steps, not an entire epoch. Both pilots use the **same Adafactor optimizer**, so the optimizer is not a hidden difference.'''),
code('''RUN_GPU_PILOT = False  # Set True after 01_preprocess_data and GPU enabled
if RUN_GPU_PILOT:
    import torch
    assert torch.cuda.is_available(), 'GPU required for timing pilot'
    for method in ['lora','full']:
        try:
            r = train_run(method, 'sib200', size=5, seed=42, rank=8, max_steps=2, pilot=True,
                          run_name=f'pilot_{method}_{int(__import__("time").time())}')
            print(method, '2 steps:', r['wall_seconds'], 'seconds;', 'VRAM GiB:', r['peak_gpu_allocated_gib'])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'{method}: OOM. Reduce sequence length / keep batch=1, or try another available GPU.')'''),
md('''## D. Runtime estimation after pilot

A rough estimate from a two-step run can be misleading because it includes model loading, initialization, and warmup. Prefer a 10–20-step pilot for better numbers. No result should be reported as a measured full-training time until that training actually completes.'''),
code('''pilots = [r for r in read_jsonl(ROOT/'outputs/metrics/train_runs.jsonl') if r.get('pilot')]
for r in pilots:
    print(r['run_id'], 'steps=',r['global_steps'], 'total wall min=',round(r['wall_seconds']/60,1),
          'avg sec/step incl. setup=',round(r['wall_seconds']/max(1,r['global_steps']),2))'''),
])

make('01_preprocess_data.ipynb', '01 — Dataset download, validation, and few-shot preparation',
     'Download the official SIB-200 English/Urdu splits, preserve source IDs, detect leakage, and create repeatable nested training subsets.', [
md('''**Rules:** Do not alter or resplit the official test data; do not use the paper's 1,000 test headlines for training. Keep Urdu and English labels identical, save source IDs, and verify that no corresponding sentence appears in different splits. Datasets are downloaded from [Davlan/SIB-200](https://huggingface.co/datasets/Davlan/sib200).'''),
code('''from datasets import load_dataset
from collections import Counter
TASK = 'sib200'
LANGS = {'ur': 'urd_Arab','en':'eng_Latn'}
DATA = {}
for lang, lang_id in LANGS.items():
    DATA[lang] = load_dataset('Davlan/sib200', lang_id)
    print(lang, {s: len(DATA[lang][s]) for s in DATA[lang]})
    print('fields:',DATA[lang]['train'].column_names)
    print('label feature:',DATA[lang]['train'].features['label'])'''),
code('''labels = list(DATA['ur']['train'].features['label'].names)
assert labels == DEFAULT_LABELS, f'Unexpected label ordering: {labels}; inspect before proceeding.'
assert list(DATA['en']['train'].features['label'].names) == labels, 'English/Urdu labels differ'
rows = []
for lang, dsd in DATA.items():
    for split in ['train','validation','test']:
        for i, r in enumerate(dsd[split]):
            item = dict(example_id=stable_id(TASK, lang, split, str(r['index_id'])),
                        index_id=str(r['index_id']), source='Davlan/sib200', lang=lang, split=split,
                        text=str(r['text']), label_id=int(r['label']), label=labels[int(r['label'])])
            rows.append(item)
problems=validate_split(rows,labels)
print('Rows:',len(rows), 'issues:',problems[:10])
assert not problems, 'Inspect split leakage, missing values or duplicate IDs before proceeding.'
for s in ['train','validation','test']:
    en = {r['index_id']: r['label_id'] for r in rows if r['lang']=='en' and r['split']==s}
    ur = {r['index_id']: r['label_id'] for r in rows if r['lang']=='ur' and r['split']==s}
    assert en==ur, 'Parallel English and Urdu IDs/labels do not match; inspect before continuing.'
print('Parallel languages and labels match by index_id across all splits.')'''),
code('''out = ROOT/'data'/TASK; out.mkdir(exist_ok=True)
for lang in LANGS:
    for split in ['train','validation','test']:
        group = [r for r in rows if r['lang']==lang and r['split']==split]
        write_csv(out / f'{lang}_{split}.csv', group)
counts = {lang: {s:Counter(r['label'] for r in rows if r['lang']==lang and r['split']==s)
                 for s in ['train','validation','test']} for lang in LANGS}
manifest = {'task':TASK, 'dataset':'Davlan/sib200','language_configs':LANGS, 'labels':labels,
            'rows_by_split': {f'{lang}_{split}':len([r for r in rows if r['lang']==lang and r['split']==split])
                              for lang in LANGS for split in ['train','validation','test']},
            'sha256':{f'{lang}_{split}':file_hash(out/f'{lang}_{split}.csv')
                      for lang in LANGS for split in ['train','validation','test']},
            'notes':'Fixed official splits; class mapping taken from downloaded ClassLabel features. Do not change test labels.'}
write_json(out/'manifest.json',manifest)
print('Saved:',out)
for lang in LANGS:
    print(lang,'class counts:',{split:dict(c) for split,c in counts[lang].items()})'''),
code('''for seed in [42,123,456]:
    subsets = stratified_prefix_samples(load_task(TASK,'ur','train'), [5,10,25,50,100], seed=seed)
    for n, entry in subsets.items():
        if entry['available']:
            write_csv(out / f'ur_train_{n}perclass_seed{seed}.csv', entry['rows'])
            print(f'seed={seed} n/class={n} total={len(entry["rows"])}')
        else:
            print(f'seed={seed} n/class={n} NOT available:', entry['shortfall'])
print('Important: 100 examples per class is not guaranteed; do not oversample without explicitly reporting it.')'''),
])

make('02_zero_fewshot_baselines.ipynb', '02 — Zero-shot and few-shot baselines',
     'Evaluate the unchanged Qwen model in Urdu and English; few-shot demonstrations come ONLY from a training split.', [
md('''Run `01_preprocess_data` first. Zero-shot means zero labeled examples in the prompt. Few-shot uses selected **training** examples as demonstrations; these are not gradient updates. Keep greedy decoding and the same exact allowed labels throughout all experiments.'''),
code('''from collections import Counter
BASE_MODEL = cfg()['model_id']
EVAL_SPLIT = 'validation'  # use test only after deciding hyperparameters
LANG = 'ur'
N_FEWSHOT_PER_CLASS = 1
MAX_ROWS = None  # use 20 for a smoke run, None for complete split
labels = task_labels('sib200')
rows = load_task('sib200', LANG, EVAL_SPLIT)
if MAX_ROWS is not None: rows = rows[:MAX_ROWS]
print('Rows:',len(rows),'Labels:',labels)
model, tokenizer = load_model(BASE_MODEL)
print('Model loaded.')'''),
code('''demonstrations = examples_for('sib200', N_FEWSHOT_PER_CLASS, 42, lang=LANG)
assert all(r['split']=='train' for r in demonstrations)
print('Demonstrations from training only:',Counter(r['label'] for r in demonstrations))
for condition, demos in [('zero_shot',None),('few_shot',demonstrations)]:
    output = infer_rows(model, tokenizer, rows, labels, LANG, cfg()['max_new_tokens'], demos)
    path = ROOT/'outputs/predictions'/f'sib200_{condition}_{LANG}_{EVAL_SPLIT}.csv'
    meta = dict(task='sib200',lang=LANG,split=EVAL_SPLIT,run_id=condition,model_id=BASE_MODEL,
                dataset_sha256=dataset_sha('sib200',LANG,EVAL_SPLIT),
                subset=(MAX_ROWS is not None),n_fewshot_per_class=N_FEWSHOT_PER_CLASS if demos else 0)
    save_prediction_run(path,output,meta)
    print(condition, len(output), 'saved to',path, 'valid',sum(x['valid'] for x in output))
    print('First five:',[(r['reference'],r['prediction'],r['raw_prediction']) for r in output[:5]])'''),
md('''When done, release GPU memory before training. **Do not use validation performance as published final test performance.**'''),
code('''del model
import torch
if torch.cuda.is_available(): torch.cuda.empty_cache()'''),
])

make('03_train_lora.ipynb','03 — LoRA adaptation',
     'Train adapter-only versions of the same Qwen checkpoint; save logs, trained-parameter count, dataset hashes and adapter files.', [
md('''### Training choices

Default: `10 examples/class`, rank 8, seed 42, two epochs. After the pilot works, add `25,50` and seed repeats. All training uses only the training split. Validation is available for selection but the shared training function does not read the test set.'''),
code('''import torch
assert torch.cuda.is_available(), 'Enable a GPU on Kaggle/Colab. CPU fine-tuning is intentionally blocked here.'
TASK='sib200'; LANG='ur'
SIZES=[10]  # after pilot, [10,25,50]
RANKS=[8]  # after pilot, [8,16] at a fixed training size
SEEDS=[42] # after pilot, [42,123,456]
MODEL_OVERRIDE = None  # set to saved English anchor path for controlled forgetting
DRY_RUN = False
print('GPU:',torch.cuda.get_device_name(0))
for seed in SEEDS:
    for n in SIZES:
        for rank in RANKS:
            run_name = f'{TASK}_lora_{LANG}_n{n}_s{seed}_r{rank}' + ('_anchor' if MODEL_OVERRIDE else '')
            if (ROOT/'outputs/checkpoints'/run_name/'complete.json').exists():
                print('Already complete; skip:',run_name); continue
            if DRY_RUN: print('Would run:',run_name); continue
            r=train_run('lora',TASK,n,seed,run_name=run_name,lang=LANG,rank=rank,model_override=MODEL_OVERRIDE)
            print('FINISHED',r['run_id'],'minutes:',round(r['wall_seconds']/60,2),
                  'peak GiB:',r['peak_gpu_allocated_gib'],'trainable %:',r['trainable_pct'])'''),
md('''The checkpoint contains **LoRA adapters**, not a standalone copy of the full Qwen model. For inference the base model and adapter must both be loaded. Do not combine adapters from different base checkpoints.'''),
])

make('04_train_full.ipynb','04 — Full model fine-tuning',
     'Update all model parameters using the same examples, chat loss, optimizer family, and evaluation rules as LoRA.', [
md('''**IMPORTANT MEMORY NOTE:** Full training of 0.5B requires model weights, gradients, and optimizer state. Adafactor is selected **for both LoRA and full** to reduce optimizer-memory pressure and keep the optimizer family controlled. It is *not* guaranteed to fit T4 in every session. Start with the 2-step pilot in notebook 00 and batch size 1.'''),
code('''import torch
assert torch.cuda.is_available(), 'Enable a GPU'
TASK='sib200'; LANG='ur'; SIZES=[10]; SEEDS=[42]
MODEL_OVERRIDE=None # optionally an English-trained merged checkpoint, same as LoRA notebook
DRY_RUN=False
for seed in SEEDS:
    for n in SIZES:
        run_name=f'{TASK}_full_{LANG}_n{n}_s{seed}' + ('_anchor' if MODEL_OVERRIDE else '')
        if (ROOT/'outputs/checkpoints'/run_name/'complete.json').exists():
            print('Already complete:',run_name);continue
        if DRY_RUN: print('Would run',run_name); continue
        try:
            r=train_run('full',TASK,n,seed,run_name=run_name,lang=LANG,model_override=MODEL_OVERRIDE)
            print('FINISHED',r['run_id'],'minutes:',round(r['wall_seconds']/60,2),'GPU GiB:',r['peak_gpu_allocated_gib'])
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print('FULL TRAIN OOM. Stop this kernel; reduce max_length to 128 or use more memory.')
            print('Do not claim a full-tuning result until a completed checkpoint is saved.')
            raise'''),
md('''When you have successful results at 10/class, add 25 and 50/class. Compare the same sizes and seeds as LoRA. Learning rates can differ between methods, but tune them fairly using the validation set, then freeze settings before final test evaluation.'''),
])

make('05_inference.ipynb','05 — Inference and prediction saving',
     'Load a saved LoRA adapter or a fully tuned model, generate one category per example, and save raw outputs and metadata.', [
md('''Run once per trained checkpoint and language. **Only change `SPLIT` to `test` after finalizing training settings.** Run the English test on the same checkpoint if examining retention. The baseline model may use this notebook with `METHOD="base"`.'''),
code('''import torch
TASK='sib200'; LANG='ur'; SPLIT='validation'  # finally: 'test'
METHOD='lora'  # 'lora', 'full', or 'base'
RUN_ID='sib200_lora_ur_n10_s42_r8'
MAX_ROWS=None  # 10 for smoke, None for full split
labels = task_labels(TASK)
rows = load_task(TASK,LANG,SPLIT)
if MAX_ROWS is not None: rows=rows[:MAX_ROWS]
model_id=cfg()['model_id']; checkpoint=None
if METHOD!='base':
    state=read_json(ROOT/'outputs/checkpoints'/RUN_ID/'complete.json')
    if state['method'] != METHOD or state['task'] != TASK: raise ValueError('Checkpoint metadata does not match method/task')
    model_id=state['base_model']; checkpoint=Path(state['saved_path'])
    if not checkpoint.exists(): checkpoint=ROOT/'outputs/checkpoints'/RUN_ID/'model'
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    if METHOD=='full':
        model,tokenizer=load_model(str(checkpoint))
    else:
        model,tokenizer=load_model(model_id,adapter_path=str(checkpoint))
else:
    model,tokenizer=load_model(model_id)
print('Model:',model_id,'mode:',METHOD,'rows:',len(rows))'''),
code('''path=ROOT/'outputs/predictions'/f'{RUN_ID}_{LANG}_{SPLIT}.csv'
meta=dict(task=TASK,lang=LANG,split=SPLIT,model_id=model_id,
          run_id=RUN_ID,method=METHOD,checkpoint=str(checkpoint) if checkpoint else None,
          dataset_sha256=dataset_sha(TASK,LANG,SPLIT),subset=(MAX_ROWS is not None),
          decoding={'do_sample':False,'max_new_tokens':cfg()['max_new_tokens']})
outputs=infer_resume(model,tokenizer,rows,labels,LANG,path,meta,cfg()['max_new_tokens'])
print('Saved:',path, 'count:',len(outputs),'invalid:',sum(not x['valid'] for x in outputs))
print('First outputs:',[(p['reference'],p['prediction'],p['raw_prediction']) for p in outputs[:8]])'''),
code('''del model
if torch.cuda.is_available(): torch.cuda.empty_cache()'''),
])

make('06_evaluate.ipynb', '06 — Scientific evaluation',
     'Validate prediction coverage, source hashes and label set before reporting macro-F1, accuracy and invalid-output rates.', [
md('''This notebook is deliberately strict. It rejects predictions for the wrong task, partial samples, wrong dataset versions, unknown example IDs and duplicate IDs. Invalid generated labels are counted as mistakes, not silently discarded.'''),
code('''import pandas as pd
TASK='sib200'; LANG='ur'; SPLIT='validation'  # switch to 'test' only at final evaluation
paths=sorted((ROOT/'outputs/predictions').glob(f'*_{LANG}_{SPLIT}.csv'))
print('Found',len(paths),'prediction files')
for p in paths: print(p.name)'''),
code('''results=[]
rows=load_task(TASK,LANG,SPLIT); labels=task_labels(TASK)
for p in paths:
    meta_path=p.with_suffix('.meta.json')
    if not meta_path.exists(): print('SKIP (no metadata):',p); continue
    meta=read_json(meta_path)
    if meta['task']!=TASK: print('SKIP other task:',p.name); continue
    try:
        validate_prediction_metadata(meta,TASK,LANG,SPLIT)
        if meta.get('subset'): raise ValueError('partial/smoke prediction set cannot be called full evaluation')
        predictions=read_csv(p)
        score=evaluate_predictions(rows,predictions,labels)
        compact={k:v for k,v in score.items() if k not in ['classification_report','confusion_matrix','confusion_labels']}
        compact.update({'run_id':meta['run_id'],'method':meta.get('method',meta['run_id']),
                        'task':TASK,'lang':LANG,'split':SPLIT,'file':str(p)})
        results.append(compact)
        write_json(ROOT/'outputs/metrics'/f'{p.stem}_metrics.json',dict(**compact, details=score, metadata=meta))
        print(p.name,'accuracy',round(score['accuracy'],4),'macro-F1',round(score['macro_f1'],4),
              'invalid',score['invalid_count'])
    except (ValueError,KeyError,FileNotFoundError) as e:
        print('REJECTED',p.name,'reason:',e)
if results:
    df=pd.DataFrame(results).sort_values('macro_f1',ascending=False)
    display(df[['run_id','split','count','accuracy','macro_f1','invalid_count']])
    df.to_csv(ROOT/'outputs/metrics'/f'{TASK}_{LANG}_{SPLIT}_leaderboard.csv',index=False)
else: print('No complete, verified predictions available yet.')'''),
])

make('07_results_analysis.ipynb', '07 — Results, plots, errors and runtime analysis',
     'Create report-ready tables and charts using only completed and validated training and inference records.', [
md('''This notebook does not invent missing metrics or training times. Plots appear only when result files exist. **Do not mix validation and test metrics**, or compare different task label sets as if they were the same experiment.'''),
code('''import pandas as pd, numpy as np, matplotlib.pyplot as plt
from collections import Counter
RUNS=read_jsonl(ROOT/'outputs/metrics/train_runs.jsonl')
runs=pd.DataFrame([x for x in RUNS if not x.get('pilot') and x.get('status')=='complete'])
if len(runs):
    display(runs[['run_id','method','examples','lora_rank','seed','trainable_pct','wall_seconds','peak_gpu_allocated_gib']])
else: print('No completed training results yet.')'''),
code('''paths=sorted((ROOT/'outputs/metrics').glob('*_leaderboard.csv'))
if paths:
    scores=pd.concat([pd.read_csv(x) for x in paths],ignore_index=True)
    display(scores[['run_id','lang','split','accuracy','macro_f1','invalid_count']])
    for key,group in scores.groupby(['task','lang','split']):
        ax=group.plot(kind='bar',x='run_id',y='macro_f1',legend=False,figsize=(max(7,len(group)*0.8),4))
        ax.set_title('Macro-F1 | '+str(key));ax.set_ylim(0,1);ax.set_ylabel('Macro-F1');
        plt.xticks(rotation=35,ha='right');plt.tight_layout()
        plt.savefig(ROOT/'outputs/plots'/('_'.join(key)+'_macro_f1.png'),dpi=160); plt.show()
else: print('No evaluated metrics yet. Run 05 then 06.')'''),
md('''## Paired prediction comparison

For a scientific comparison, models must be evaluated on the same test examples. This cell prints correct/incorrect disagreements for a pair. It does not claim significance from one seed or a tiny test set.'''),
code('''COMPARE_A=None # example: 'sib200_lora_ur_n10_s42_r8_ur_test.csv'
COMPARE_B=None # example: 'sib200_full_ur_n10_s42_ur_test.csv'
if COMPARE_A and COMPARE_B:
    pa=ROOT/'outputs/predictions'/COMPARE_A; pb=ROOT/'outputs/predictions'/COMPARE_B
    ma,mb=read_json(pa.with_suffix('.meta.json')),read_json(pb.with_suffix('.meta.json'))
    if (ma['task'],ma['lang'],ma['split'],ma['dataset_sha256']) != (mb['task'],mb['lang'],mb['split'],mb['dataset_sha256']):
        raise ValueError('Different tasks/languages/splits/dataset hashes')
    da=pd.DataFrame(read_csv(pa)).set_index('example_id');db=pd.DataFrame(read_csv(pb)).set_index('example_id')
    assert set(da.index)==set(db.index),'Incomplete example alignment'
    aligned=da.join(db,lsuffix='_A',rsuffix='_B')
    a=(aligned['prediction_A']==aligned['reference_A'])
    b=(aligned['prediction_B']==aligned['reference_B'])
    print('Only A correct:',int((a & ~b).sum()), 'Only B correct:',int((~a & b).sum()))
    diff=aligned[a!=b]
    display(diff.head(30))
else: print('Set COMPARE_A/COMPARE_B to two complete prediction files.')'''),
md('''## Examine incorrectly classified Urdu examples

Review 20–50 errors manually for topic ambiguity, input length, misunderstood words, and invalid labels.'''),
code('''ERROR_PREDICTION_FILE=None
if ERROR_PREDICTION_FILE:
    p=ROOT/'outputs/predictions'/ERROR_PREDICTION_FILE
    meta=read_json(p.with_suffix('.meta.json'))
    validate_prediction_metadata(meta,meta['task'],meta['lang'],meta['split'])
    reference=pd.DataFrame(load_task(meta['task'],meta['lang'],meta['split']))
    predicted=pd.DataFrame(read_csv(p))
    merged=predicted.merge(reference[['example_id','text','label']],on='example_id',validate='one_to_one')
    errors=merged[merged.prediction!=merged.label]
    errors[['example_id','text','label','prediction','raw_prediction']].head(50).to_csv(ROOT/'outputs/metrics'/'error_analysis_sample.csv',index=False)
    display(errors[['text','label','prediction','raw_prediction']].head(30))
else: print('Choose a validated inference file.')'''),
md('''## Actual GPU duration and memory (not predicted numbers)'''),
code('''if len(runs):
    f=runs.copy(); f['minutes']=f.wall_seconds/60
    ax=f.plot(kind='bar',x='run_id',y='minutes',legend=False,figsize=(max(7,len(f)*0.8),4))
    ax.set_ylabel('Actual completed training minutes');plt.xticks(rotation=35,ha='right');plt.tight_layout()
    plt.savefig(ROOT/'outputs/plots'/'actual_train_times.png',dpi=160);plt.show()
else: print('Training time graphs require completed runs.')'''),
])

make('08_paper_benchmark_audit.ipynb', '08 — Published EMNLP Urdu benchmark audit',
     'Verify the exact original eight-category test set; prepare a separate task only if independent eight-category TRAINING data exists.', [
md('''**Critical:** The [paper repository](https://github.com/sameearif/Generalists-vs-Specialists) gives the *test* data under `datasets/topic-classification.csv` with columns `input,label`. It has **8 labels**, unlike SIB-200's 7. Do **not** train on the paper test set or map SIB-200 labels into these eight categories. Published 0.8474 macro-F1 applies to the paper's test set, not SIB-200.'''),
code('''# Download the public repository. Kaggle Internet or Colab Internet must be enabled.
import subprocess
RUN_CLONE=False
repo_dir=ROOT/'data/external/Generalists-vs-Specialists'
repo_dir.parent.mkdir(parents=True,exist_ok=True)
if RUN_CLONE and not repo_dir.exists():
    subprocess.check_call(['git','clone','--depth','1',cfg()['paper_repo_url'],str(repo_dir)])
paper_test=repo_dir/'datasets/topic-classification.csv'
if paper_test.exists():
    items=read_csv(paper_test)
    print('Paper test rows:',len(items),'columns:',list(items[0]))
    print('Test label counts:',__import__('collections').Counter(r['label'] for r in items))
else:
    print('Clone repo first, then re-run. Expected:',paper_test)'''),
code('''# Audit, not training. Every label ID must map to the repository's documented label_mapping.py.
if paper_test.exists():
    test_raw=read_csv(paper_test)
    assert set(test_raw[0]) >= {'input','label'}
    assert all(str(r['label']).strip() in [str(x) for x in range(8)] for r in test_raw)
    print('SIB-200 labels:',DEFAULT_LABELS)
    print('Published paper 8 labels:',PAPER_LABELS)
    print('Same class set?',set(DEFAULT_LABELS)==set(PAPER_LABELS))
    print('Paper test is held out and MUST NOT be used for training/validation.')'''),
md('''### Optional 8-class reproduction

Provide a **separate** `paper8_train.csv` and `paper8_validation.csv` with `text,label` columns, using *exactly* the eight published label strings. These files must come from a licensed training source that you document. No rows from the published test CSV may appear in them. If no legitimate train/validation set can be found, stop here and report the published score as related work only.'''),
code('''TRAIN_FILE=ROOT/'data/external/paper8_train.csv'
VALID_FILE=ROOT/'data/external/paper8_validation.csv'
RUN_PAPER8_IMPORT=False
if RUN_PAPER8_IMPORT:
    assert paper_test.exists() and TRAIN_FILE.exists() and VALID_FILE.exists(), 'Missing paper test or independent training data'
    train, val, test = read_csv(TRAIN_FILE),read_csv(VALID_FILE),read_csv(paper_test)
    assert all(set(r)>= {'text','label'} and r['label'] in PAPER_LABELS for r in train+val)
    test_rows=[]
    for i,r in enumerate(test):
        label=PAPER_LABELS[int(r['label'])]
        test_rows.append({'example_id':stable_id('paper8','ur','test',i),'index_id':'','source':'paper EMNLP 2024','lang':'ur','split':'test','text':r['input'],'label':label,'label_id':int(r['label'])})
    def make_rows(src,split):
        return [dict(example_id=stable_id('paper8','ur',split,i),index_id='',source='independent_training',
                     lang='ur',split=split,text=r['text'],label=r['label'],label_id=PAPER_LABELS.index(r['label'])) for i,r in enumerate(src)]
    allrows=make_rows(train,'train')+make_rows(val,'validation')+test_rows
    problems=validate_split(allrows,PAPER_LABELS)
    assert not problems, f'Leakage or malformed data: {problems}'
    savedir=ROOT/'data/paper8';savedir.mkdir(parents=True,exist_ok=True)
    for split in ['train','validation','test']:
        write_csv(savedir/f'ur_{split}.csv',[x for x in allrows if x['split']==split])
    write_json(savedir/'manifest.json',dict(task='paper8',labels=PAPER_LABELS,original_test_sha256=file_hash(paper_test),
               separate_training_source='DOCUMENT YOUR TRAINING DATA HERE',
               notes='Paper eight-category test set; independently sourced training & validation; exact fixed evaluation'))
    print('Paper8 task prepared. To train: set TASK=paper8 in notebooks 03/04 and only select feasible per-class sizes.')
else: print('Paper8 import disabled; audit mode only. This avoids using held-out test data as training.')'''),
])

make('09_optional_english_anchor.ipynb', '09 — Optional English-trained anchor for forgetting',
     'Create one shared English-specialized model from which BOTH LoRA and full Urdu adaptation begin.', [
md('''A direct before/after English score from the original Qwen model is only a language-retention check, not necessarily meaningful *catastrophic forgetting*. For a stronger claim, establish English ability first using the SIB-200 English training split. Then save a **shared English-trained checkpoint**, and start both Urdu training methods from that checkpoint. The resulting comparison isolates what Urdu adaptation changes.'''),
code('''import torch
RUN_ANCHOR=False  # English training consumes additional GPU time
if RUN_ANCHOR:
    assert torch.cuda.is_available()
    r=train_run('lora','sib200',size=10,seed=42,lang='en',rank=8,
                run_name='sib200_english_anchor_lora_r8')
    print('English adapter ready:',r['saved_path'])'''),
md('''## Merge the English LoRA adapter for a common starting point

This modifies the base weights once, creating a new English-adapted checkpoint. Then both Urdu methods begin from the same saved merged model. **Do not mix the English adapter and Urdu adapter in a way that gives only one method a hidden advantage.**'''),
code('''MERGE_ANCHOR=False
if MERGE_ANCHOR:
    from peft import PeftModel
    r=read_json(ROOT/'outputs/checkpoints/sib200_english_anchor_lora_r8/complete.json')
    m,t=load_model(r['base_model'],adapter_path=r['saved_path'])
    merged=m.merge_and_unload()
    target=ROOT/'outputs/english_anchor_merged'
    target.mkdir(exist_ok=True)
    merged.save_pretrained(str(target),safe_serialization=True)
    t.save_pretrained(str(target))
    print('Saved common anchor:',target)
    print('Set MODEL_OVERRIDE=str(ROOT/"outputs/english_anchor_merged") in notebooks 03 and 04.')'''),
md('''### Cross-lingual evaluation

Use notebook 05 with `LANG='en'` and `SPLIT='test'` to compare the English anchor **before** Urdu adaptation and each Urdu-trained model **after** adaptation. Your final report should distinguish English retention from broader catastrophic forgetting.'''),
])

make('10_runtime_planner.ipynb', '10 — Pilot-based GPU time and resource planner',
     'Estimate quota needs from REAL pilot timings; report assumptions instead of pretending estimates were observed full-run durations.', [
md('''Full fine-tuning cannot be assigned one universal duration. Batch size, GPU model, token length, optimizer, and startup overhead matter. **No timing value below is a published benchmark until your own pilot runs.**'''),
code('''import math, pandas as pd
pilots=[r for r in read_jsonl(ROOT/'outputs/metrics/train_runs.jsonl') if r.get('pilot')]
if not pilots:
    print('No pilot measurements yet. Run GPU pilot in notebook 00 first.')
else:
    out=[]
    for r in pilots:
        for examples_per_class in [10,25,50]:
            n=7*examples_per_class
            effective_batch=cfg()['per_device_batch_size']*cfg()['gradient_accumulation']
            estimated_steps=math.ceil(n/effective_batch)*cfg()['train_epochs']
            # Estimate from completed actual training timings. Warmup bias is possible.
            sps=r['wall_seconds']/max(r['global_steps'],1)
            out.append(dict(pilot=r['run_id'], method=r['method'],per_class=examples_per_class,
                            approx_optimizer_steps=estimated_steps,naive_minutes=round(sps*estimated_steps/60,1),
                            warmup_included=True,estimated_from_pilot_steps=r['global_steps']))
    display(pd.DataFrame(out))
    print('Use 10–20 steps for a steadier speed measurement; 2-step pilots include heavy warmup.')'''),
md('''### Practical quota and checkpoint advice

- For 2–3-epoch runs, start at 10/class and move to 25/class only after both methods work.
- Save checkpoints after **each finished run** to persistent storage (Google Drive on Colab, Kaggle dataset/output).
- A free notebook session can disconnect; saved files in ephemeral workspace may disappear.
- If you need to preserve checkpoints through interruptions, implement step-based checkpoints and resume in a dedicated long-run notebook. This project currently defaults to saving only completed runs to reduce storage.
- A 2-step pilot alone does not measure model download time or full training duration reliably.
- Measure `nvidia-smi` and peak `torch.cuda.max_memory_allocated()`; the latter is allocated memory, not full driver-level device utilization.
'''),
])

make('11_paper8_original_headlines.ipynb','11 — Eight-class published benchmark: independent train/validation',
     'Create separate 8-class Urdu training and validation sets from the original headlines repository, excluding the published test examples.', [
md('''### Why this notebook matters

[The EMNLP 2024 paper](https://aclanthology.org/2024.findings-emnlp.426/) credits the original [Urdu-News-Headline-Dataset](https://github.com/mwaseemrandhawa/Urdu-News-Headline-Dataset), which provides Excel files for Business, Entertainment, Health, Politics, Science, Sports, Weird, and World. The published **1,000-line test CSV** is **not** training data. We remove exact matching test headlines **before** splitting any source training data. "Weird" is tentatively mapped to the paper's `other` category, but this mapping must be reviewed because the category names are not identical.

The dataset repository does not state a clear redistribution license; do not upload the original spreadsheets as part of your project without permission. Install `openpyxl` only to read the spreadsheets.'''),
code('''import subprocess, pandas as pd
SOURCE_REPO = ROOT/'data/external/Urdu-News-Headline-Dataset'
PAPER_REPO  = ROOT/'data/external/Generalists-vs-Specialists'
CLONE_REPOSITORIES = False  # Set True after enabling Internet in Kaggle/Colab
if CLONE_REPOSITORIES:
    SOURCE_REPO.parent.mkdir(parents=True,exist_ok=True)
    for url,path in [('https://github.com/mwaseemrandhawa/Urdu-News-Headline-Dataset',SOURCE_REPO),
                     ('https://github.com/sameearif/Generalists-vs-Specialists',PAPER_REPO)]:
        if not path.exists(): subprocess.check_call(['git','clone','--depth','1',url,str(path)])
print('Original source files:',[p.name for p in SOURCE_REPO.glob('*.xlsx')])
print('Published test:',PAPER_REPO/'datasets/topic-classification.csv')'''),
code('''# INSPECT before importing. Avoid guessing the headline column in the source XLSX files.
example_file=next(iter(SOURCE_REPO.glob('*.xlsx')),None)
if example_file:
    table=pd.read_excel(example_file,nrows=5)
    print('Example category file:',example_file.name)
    print('Available columns:',list(table.columns))
    display(table.head(3))
else: print('Clone source repository and ensure openpyxl is installed.')'''),
md('''### Explicit source-field mapping and safety gates

Set `HEADLINE_COLUMN` to the column containing the actual headline after inspecting the previous output. For example, it might be `Title`, but do **not** assume this is true. Set `CONFIRMED_CATEGORY_MAPPING=True` only after confirming that `Weird.xlsx` corresponds to the paper's `other` class.'''),
code('''HEADLINE_COLUMN=None  # exact column name based on inspection above
CONFIRMED_CATEGORY_MAPPING=False  # review Weird -> other first
BUILD_DATA=False
MAX_TRAIN_PER_CLASS=500
MAX_VAL_PER_CLASS=50
SEED=42
CAT_FILES={'Business':'business','Entertainment':'entertainment','Health':'health',
           'Politics':'politics','Science':'science','Sports':'sports','Weird':'other','World':'world'}
print('Paper class order:',PAPER_LABELS)'''),
code('''import re,unicodedata,random,hashlib
from collections import Counter

def fingerprint(text):
    text=unicodedata.normalize('NFKC',str(text)).casefold()
    text=re.sub(r'[\u200b-\u200f\ufeff]','',text)
    return re.sub(r'\\s+',' ',text).strip()

if BUILD_DATA:
    assert HEADLINE_COLUMN and CONFIRMED_CATEGORY_MAPPING, 'Review columns and class map before starting.'
    test_file=PAPER_REPO/'datasets/topic-classification.csv'
    assert test_file.exists(), 'Run notebook 08 or clone the paper repo'
    paper_test=read_csv(test_file)
    assert len(paper_test)==1000 and set(paper_test[0])>={'input','label'}
    test_fingerprints={fingerprint(r['input']) for r in paper_test}
    sources=[]
    for fname,label in CAT_FILES.items():
        file=SOURCE_REPO/f'{fname}.xlsx'
        assert file.exists(),f'Missing {file}'
        source=pd.read_excel(file)
        assert HEADLINE_COLUMN in source, f'{HEADLINE_COLUMN} missing in {file.name}, columns={list(source.columns)}'
        extracted=source[HEADLINE_COLUMN].dropna().astype(str).tolist()
        print(file.name,'rows:',len(extracted))
        for text in extracted:
            if len(fingerprint(text))>3: sources.append((fingerprint(text),text,label))
    # Reject duplicate source headlines across classes, not just within one class.
    by_fingerprint={}
    for fp,txt,label in sources: by_fingerprint.setdefault(fp,set()).add(label)
    ambiguous={fp for fp,labset in by_fingerprint.items() if len(labset)>1}
    unique={}
    for fp,txt,label in sources:
        if fp in test_fingerprints or fp in ambiguous: continue
        unique.setdefault(fp,(txt,label))
    print('Discarded test overlaps (unique):',len({fp for fp,_,_ in sources if fp in test_fingerprints}))
    print('Discarded same-headline conflicting classes:',len(ambiguous))
    grouped={label:[] for label in PAPER_LABELS}
    for fp,(txt,label) in unique.items(): grouped[label].append((fp,txt))
    train,val=[],[]
    for label in PAPER_LABELS:
        group=sorted(grouped[label]); random.Random(f'{SEED}:{label}').shuffle(group)
        assert len(group)>=MAX_TRAIN_PER_CLASS+MAX_VAL_PER_CLASS, f'Not enough safe {label} rows: {len(group)}'
        chosen_val=group[:MAX_VAL_PER_CLASS]
        chosen_train=group[MAX_VAL_PER_CLASS:MAX_VAL_PER_CLASS+MAX_TRAIN_PER_CLASS]
        for split,selected,target in [('train',chosen_train,train),('validation',chosen_val,val)]:
            for fp,txt in selected:
                target.append(dict(example_id=stable_id('paper8','ur',split,hashlib.sha256(fp.encode()).hexdigest()[:16]),
                                   index_id='',source='original headlines repo',lang='ur',split=split,
                                   text=txt,label=label,label_id=PAPER_LABELS.index(label)))
    test=[]
    for i,r in enumerate(paper_test):
        v=int(r['label'])
        test.append(dict(example_id=stable_id('paper8','ur','test',i),index_id='',
                         source='EMNLP2024 held-out evaluation',lang='ur',split='test',
                         text=r['input'],label=PAPER_LABELS[v],label_id=v))
    assert not validate_split(train+val+test,PAPER_LABELS)
    out=ROOT/'data/paper8';out.mkdir(parents=True,exist_ok=True)
    for name,rs in [('train',train),('validation',val),('test',test)]:
        write_csv(out/f'ur_{name}.csv',rs)
    write_json(out/'manifest.json',dict(task='paper8',labels=PAPER_LABELS,source_repository='mwaseemrandhawa/Urdu-News-Headline-Dataset',
               paper_test_sha256=file_hash(test_file),seed=SEED,
               notes='Source classes checked; exact text deduplicated; 8-class original paper test.'))
    print('READY for 03/04/05 with TASK=paper8 and counts:',len(train),len(val),len(test))
else:
    print('Data build disabled until you inspect the spreadsheet column and confirm the class mapping.')'''),
md('''**Remaining limitation:** Exact-match filtering does not prove that near-duplicate headlines or translated versions are absent. Inspect high-similarity train/test headlines before claiming no contamination. For a paper comparison, retain the exact 1,000 test inputs and the same macro-F1 calculation; training settings differ from the published 8B experiment and must be disclosed.'''),
])

print('Generated',len(list(N.glob('*.ipynb'))),'notebooks')
