import nbformat as nbf
from pathlib import Path
path=Path(__file__).resolve().parent/'notebooks/12_uncertainty_and_retention.ipynb'
md=lambda s: nbf.v4.new_markdown_cell(s)
cd=lambda s: nbf.v4.new_code_cell(s)
nb=nbf.v4.new_notebook(cells=[
md('''# 12 — Uncertainty, paired model comparison, and English retention

**Purpose:** Compare two complete model runs on exactly the same test items. Bootstrap confidence intervals describe uncertainty from limited test samples, not randomness from retraining. For robust conclusions, also use several training seeds.'''),
cd('''from pathlib import Path
import os,sys
candidates=[Path.cwd(),Path.cwd().parent,Path('/kaggle/working/Urdu_Qwen_Research_Notebooks'),Path('/content/Urdu_Qwen_Research_Notebooks')]
ROOT=next((p for p in candidates if (p/'project_utils.py').exists()),None)
if ROOT is None: raise FileNotFoundError('Extract the whole project ZIP; see README.')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from project_utils import *
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import f1_score'''),
md('''## A. Paired bootstrap for LoRA versus full

Supply two complete prediction files from notebook 05 and make sure their metadata refer to the same dataset, split, and language. Invalid generated labels must remain wrong answers.'''),
cd('''TASK='sib200';LANG='ur';SPLIT='test'
FILE_A=None # example: sib200_lora_ur_n10_s42_r8_ur_test.csv
FILE_B=None # example: sib200_full_ur_n10_s42_ur_test.csv
N_BOOTSTRAP=1000
if FILE_A and FILE_B:
    pa,pb=[ROOT/'outputs/predictions'/name for name in [FILE_A,FILE_B]]
    ma,mb=read_json(pa.with_suffix('.meta.json')),read_json(pb.with_suffix('.meta.json'))
    validate_prediction_metadata(ma,TASK,LANG,SPLIT)
    validate_prediction_metadata(mb,TASK,LANG,SPLIT)
    assert not ma.get('subset') and not mb.get('subset')
    da=pd.DataFrame(read_csv(pa)).set_index('example_id').sort_index()
    db=pd.DataFrame(read_csv(pb)).set_index('example_id').sort_index()
    assert da.index.tolist()==db.index.tolist() and (da.reference==db.reference).all()
    labels=task_labels(TASK)
    y=da.reference.to_numpy();a=da.prediction.to_numpy();b=db.prediction.to_numpy()
    def score(ii,p): return f1_score(y[ii],p[ii],labels=labels,average='macro',zero_division=0)
    indices=np.arange(len(y));observed=score(indices,a)-score(indices,b)
    rng=np.random.default_rng(42)
    deltas=[score(ii:=rng.integers(0,len(y),len(y)),a)-score(ii,b) for _ in range(N_BOOTSTRAP)]
    low,high=np.percentile(deltas,[2.5,97.5])
    print('A minus B macro-F1:',round(observed,4))
    print('Paired bootstrap 95% interval:',round(low,4),round(high,4))
    print('TRAINING SEED VARIATION IS NOT INCLUDED: run 3 seeds to study training randomness.')
else:
    print('Set FILE_A and FILE_B to matching, complete prediction CSVs.')'''),
md('''## B. English performance retention

To investigate forgetting properly, evaluate the original English-trained anchor **before** Urdu training, and evaluate the Urdu-adapted LoRA and full models on the unchanged English test. The same labels and exact examples must be used.'''),
cd('''ENGLISH_BEFORE=None # e.g. outputs/metrics/<name>_metrics.json
ENGLISH_AFTER_LORA=None
ENGLISH_AFTER_FULL=None
if all([ENGLISH_BEFORE,ENGLISH_AFTER_LORA,ENGLISH_AFTER_FULL]):
    b,l,f=[read_json(ROOT/x) for x in [ENGLISH_BEFORE,ENGLISH_AFTER_LORA,ENGLISH_AFTER_FULL]]
    for x in [b,l,f]: assert x['task']=='sib200' and x['lang']=='en' and x['split']=='test'
    print('Before Urdu fine-tuning:',round(b['macro_f1'],4))
    print('English F1 change with Urdu LoRA:',round(l['macro_f1']-b['macro_f1'],4))
    print('English F1 change with Urdu full:',round(f['macro_f1']-b['macro_f1'],4))
else: print('Select all three English test metric JSON files.')'''),
md('''## C. Confusion matrix

This plot includes an additional INVALID category, so the classifier's formatting mistakes cannot disappear from the analysis.'''),
cd('''METRICS_FILE=None
if METRICS_FILE:
    record=read_json(ROOT/METRICS_FILE)
    cm=np.array(record['details']['confusion_matrix']); labels=record['details']['confusion_labels']
    plt.figure(figsize=(10,8));plt.imshow(cm,cmap='Blues')
    plt.xticks(range(len(labels)),labels,rotation=45,ha='right')
    plt.yticks(range(len(labels)),labels);plt.colorbar()
    plt.xlabel('Predicted');plt.ylabel('True');plt.title(record['run_id'])
    plt.tight_layout();plt.savefig(ROOT/'outputs/plots/confusion_matrix.png',dpi=170);plt.show()
else: print('Select a completed metrics JSON file.')''')
], metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbf.validate(nb);nbf.write(nb,path)
print('Saved',path)
