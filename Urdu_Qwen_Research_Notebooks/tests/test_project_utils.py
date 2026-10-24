import json
from pathlib import Path
from collections import Counter
import pytest
from project_utils import (DEFAULT_LABELS, PAPER_LABELS, normalize_label, stratified_prefix_samples,
                           validate_split, encode_supervised, evaluate_predictions, messages_for,
                           stable_id, file_hash, write_json, read_json)


class ToyChatTokenizer:
    """Fake tokenizer with the exact same prefix property expected of an instruct LLM."""
    def encode(self, text, add_special_tokens=False):
        return [ord(x) for x in text]
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(x) for x in ids)
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages)
        if add_generation_prompt:
            rendered += '<assistant>'
        return [ord(ch) for ch in rendered]


def make_rows():
    return [dict(example_id=stable_id('toy','ur','train',i),lang='ur',split='train',
                 label='sports' if i%2 else 'politics',text=f'unique sample {i}',index_id=str(i)) for i in range(20)]


def test_strict_labels():
    assert normalize_label('  SPORTS.  ', DEFAULT_LABELS)=='sports'
    assert normalize_label('science/technology', DEFAULT_LABELS)=='science/technology'
    assert normalize_label('sports and politics', DEFAULT_LABELS) is None
    assert normalize_label('Here is the label: sports', DEFAULT_LABELS) is None
    assert normalize_label('science', DEFAULT_LABELS) is None
    assert normalize_label('world', DEFAULT_LABELS) is None
    assert len(PAPER_LABELS)==8 and len(DEFAULT_LABELS)==7


def test_no_replacement_and_nested_sampling():
    rows=make_rows()
    result=stratified_prefix_samples(rows,[2,5,11],42)
    assert len(result[2]['rows'])==4
    assert len(result[5]['rows'])==10
    assert not result[11]['available']
    ids2={r['example_id'] for r in result[2]['rows']}
    ids5={r['example_id'] for r in result[5]['rows']}
    assert ids2 <= ids5
    assert result[5]['rows']==stratified_prefix_samples(rows,[5],42)[5]['rows']
    assert Counter(r['label'] for r in result[5]['rows']) == {'politics':5,'sports':5}


def test_cross_split_text_leakage():
    a=make_rows()
    b=dict(a[0]); b['example_id']='new'; b['split']='test'; b['index_id']='100'
    assert any('text across splits' in x for x in validate_split(a+[b],['politics','sports']))


def test_cross_split_id_leakage():
    a=make_rows()
    b=dict(a[0]);b['example_id']='new';b['split']='test'; b['text']='different'
    assert any('index_id across splits' in x for x in validate_split(a+[b],['politics','sports']))


def test_supervision_masks_prompt():
    tokenizer=ToyChatTokenizer()
    item=encode_supervised(tokenizer,'urdu test','sports',['sports','politics'],'ur',2048)
    assert item['labels'].count(-100)>0
    assert item['labels'][-1] != -100
    assert len(item['labels']) == len(item['input_ids'])
    assert item['labels'].count(-100) < len(item['labels'])


def test_overlong_is_rejected_not_silent():
    with pytest.raises(ValueError,match=r'Prompt\+target alone exceeds'):
        encode_supervised(ToyChatTokenizer(),'a'*50,'sports',['sports','politics'],'en',max_length=20)


def test_strict_prediction_alignment():
    rows=[dict(example_id='a',label='sports'),dict(example_id='b',label='politics')]
    pred=[dict(example_id='a',prediction='sports'), dict(example_id='b',prediction='garbage')]
    out=evaluate_predictions(rows,pred,['sports','politics'])
    assert out['accuracy']==0.5 and out['invalid_count']==1
    assert out['macro_f1']<1
    with pytest.raises(ValueError,match='mismatched'):
        evaluate_predictions(rows,[pred[0]],['sports','politics'])
    with pytest.raises(ValueError,match='Duplicate'):
        evaluate_predictions(rows,[pred[0],pred[0]],['sports','politics'])


def test_no_test_demonstration_default():
    demos=make_rows()[:2]
    m=messages_for('other text',['sports','politics'],demonstrations=demos)
    assert len(m)==6
    assert m[1]['role']=='user' and m[2]['role']=='assistant'


def test_write_read_manifest(tmp_path):
    p=tmp_path/'sub'/'manifest.json'
    write_json(p,{'labels':['sports','politics'],'hi':'اردو'})
    assert read_json(p)['hi']=='اردو' and len(file_hash(p))==64


def test_collator_masks_padding_and_preserves_targets():
    import torch
    from project_utils import supervised_collator
    tok = type('Toy', (), {'pad_token_id':0, 'eos_token_id':2})()
    batch = supervised_collator(tok)([
        {'input_ids':[11,12,13], 'attention_mask':[1,1,1], 'labels':[-100,-100,13]},
        {'input_ids':[21,22], 'attention_mask':[1,1], 'labels':[-100,22]},
    ])
    assert batch['input_ids'].shape==(2,3)
    assert batch['attention_mask'][1].tolist()==[1,1,0]
    assert batch['labels'][1].tolist()==[-100,22,-100]


def test_inference_resume_and_metadata_protection(monkeypatch,tmp_path):
    import project_utils as m
    monkeypatch.setattr(m, 'run_generation', lambda model,tok,row,labels,lang,max_new_tokens:
                        dict(raw_prediction='sports',prediction='sports',valid=True,seconds=0.02))
    rows=[dict(example_id='a',label='sports',text='a'),dict(example_id='b',label='sports',text='b')]
    meta=dict(task='demo',lang='ur',split='test',dataset_sha256='abcdef', model_id='base',run_id='r1',
              method='lora',subset=False,decoding={'do_sample':False,'max_new_tokens':16})
    path=tmp_path/'predictions.csv'
    results=m.infer_resume(None,None,rows,['sports'],'ur',path,meta)
    assert len(results)==2 and path.exists()
    second=m.infer_resume(None,None,rows,['sports'],'ur',path,meta)
    assert len(second)==2
    bad=dict(meta);bad['model_id']='different'
    with pytest.raises(ValueError, match='different'):
        m.infer_resume(None,None,rows,['sports'],'ur',path,bad)


def test_training_truncates_source_but_not_answer():
    tk=ToyChatTokenizer()
    item=encode_supervised(tk,'a'*200,'sports',['sports','politics'],'en',max_length=350)
    assert item['source_truncated'] and item['removed_source_tokens']>0
    target=[i for i in item['labels'] if i!=-100]
    assert ''.join(chr(x) for x in target).startswith('sports')
